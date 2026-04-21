// FlashAttention-style backward (recompute P from Q,K,V and saved log-sum-exp L).
// Three launches (dQ, dK, dV) avoid atomics: each block owns one Q-tile or one KV-tile
// and sums over the complementary dimension in shared memory.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include "kernel.h"

constexpr int kBwdThreads = 128;

__host__ __device__ __forceinline__ int fa2_ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

// Default ~48 KiB/block is too small for some (Bc, Br, d) combos; opt-in to the
// device max (Blackwell / RTX 5090 supports much more per block).
__host__ static void fa2_bwd_set_max_dynamic_smem(const void* kernel, size_t bytes) {
    cudaError_t e =
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    if (e != cudaSuccess) {
        printf(
            "fa2_bwd_set_max_dynamic_smem: cudaFuncSetAttribute(%zu bytes) failed: %s\n",
            bytes,
            cudaGetErrorString(e));
    }
}

// ---------- dQ: grid (B, H, num_q_tiles); sum over all KV tiles j ----------
__global__ void fa2_bwd_dq_kernel(
    half* __restrict__ dQ,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ dO,
    const float* __restrict__ Lse,
    int Bc,
    int Br,
    int B,
    int H,
    int N,
    int d,
    int causal)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int bi = blockIdx.z;
    const int q0 = bi * Br;
    if (q0 >= N) {
        return;
    }
    const int br = min(Br, N - q0);
    const int head_stride = N * d;
    const int64_t step = (static_cast<int64_t>(b) * H + h) * head_stride;
    const int64_t Lbase = static_cast<int64_t>(b * H + h) * N;

    extern __shared__ char sh_raw[];
    half* Qi = reinterpret_cast<half*>(sh_raw);
    half* Kj = Qi + Br * d;
    half* Vj = Kj + Bc * d;
    half* dOi = Vj + Bc * d;
    const size_t half_bytes = sizeof(half) * (static_cast<size_t>(Br) * d + 2u * Bc * d + Br * d);
    float* Pbuf = reinterpret_cast<float*>(sh_raw + ((half_bytes + 15u) & ~15u));
    float* dPbuf = Pbuf + Br * Bc;
    float* sum_pdp = dPbuf + Br * Bc;
    float* dQacc = sum_pdp + Br;

    const float scale = rsqrtf(static_cast<float>(d));

    for (int idx = threadIdx.x; idx < Br * d; idx += blockDim.x) {
        dQacc[idx] = 0.f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < br * d; idx += blockDim.x) {
        const int r = idx / d;
        const int c = idx % d;
        Qi[r * d + c] = Q[step + static_cast<int64_t>(q0 + r) * d + c];
        dOi[r * d + c] = dO[step + static_cast<int64_t>(q0 + r) * d + c];
    }
    __syncthreads();

    for (int j = 0; j < N; j += Bc) {
        const int bc = min(Bc, N - j);

        if (causal && j > q0 + br - 1) {
            continue;
        }

        for (int idx = threadIdx.x; idx < bc * d; idx += blockDim.x) {
            const int r = idx / d;
            const int c = idx % d;
            const int64_t row_off = step + static_cast<int64_t>(j + r) * d;
            Kj[r * d + c] = K[row_off + c];
            Vj[r * d + c] = V[row_off + c];
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            float acc = 0.f;
            for (int k = 0; k < d; ++k) {
                acc += __half2float(Qi[r * d + k]) * __half2float(Kj[c * d + k]);
            }
            float s = acc * scale;
            if (causal) {
                const int gq = q0 + r;
                const int gk = j + c;
                if (gk > gq) {
                    s = -INFINITY;
                }
            }
            Pbuf[r * bc + c] = s;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            const float Lr = Lse[Lbase + q0 + r];
            const float s = Pbuf[r * bc + c];
            const float p = (s > -1e20f) ? expf(s - Lr) : 0.f;
            Pbuf[r * bc + c] = p;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            float acc = 0.f;
            for (int k = 0; k < d; ++k) {
                acc += __half2float(dOi[r * d + k]) * __half2float(Vj[c * d + k]);
            }
            dPbuf[r * bc + c] = acc;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < br; r += blockDim.x) {
            float sp = 0.f;
            for (int c = 0; c < bc; ++c) {
                sp += Pbuf[r * bc + c] * dPbuf[r * bc + c];
            }
            sum_pdp[r] = sp;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            const float pr = Pbuf[r * bc + c];
            const float dpr = dPbuf[r * bc + c];
            const float dsr = pr * (dpr - sum_pdp[r]);
            Pbuf[r * bc + c] = dsr;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * d; idx += blockDim.x) {
            const int r = idx / d;
            const int kk = idx % d;
            float val = 0.f;
            for (int c = 0; c < bc; ++c) {
                val += Pbuf[r * bc + c] * __half2float(Kj[c * d + kk]);
            }
            dQacc[r * d + kk] += scale * val;
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < br * d; idx += blockDim.x) {
        const int r = idx / d;
        const int kk = idx % d;
        const int64_t out_off = step + static_cast<int64_t>(q0 + r) * d + kk;
        dQ[out_off] = __float2half(dQacc[r * d + kk]);
    }
}

// ---------- dK: grid (B, H, num_kv_tiles); sum over all Q tiles i ----------
__global__ void fa2_bwd_dk_kernel(
    half* __restrict__ dK,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    const half* __restrict__ dO,
    const float* __restrict__ Lse,
    int Bc,
    int Br,
    int B,
    int H,
    int N,
    int d,
    int causal)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int bj = blockIdx.z;
    const int j0 = bj * Bc;
    if (j0 >= N) {
        return;
    }
    const int bc = min(Bc, N - j0);
    const int head_stride = N * d;
    const int64_t step = (static_cast<int64_t>(b) * H + h) * head_stride;
    const int64_t Lbase = static_cast<int64_t>(b * H + h) * N;

    extern __shared__ char sh_raw[];
    half* Kj = reinterpret_cast<half*>(sh_raw);
    half* Vj = Kj + Bc * d;
    half* Qi = Vj + Bc * d;
    half* dOi = Qi + Br * d;
    const size_t half_bytes = sizeof(half) * (2u * Bc * d + 2u * Br * d);
    float* Pbuf = reinterpret_cast<float*>(sh_raw + ((half_bytes + 15u) & ~15u));
    float* dPbuf = Pbuf + Br * Bc;
    float* sum_pdp = dPbuf + Br * Bc;
    float* dKacc = sum_pdp + Br;

    const float scale = rsqrtf(static_cast<float>(d));

    for (int idx = threadIdx.x; idx < Bc * d; idx += blockDim.x) {
        dKacc[idx] = 0.f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < bc * d; idx += blockDim.x) {
        const int r = idx / d;
        const int c = idx % d;
        const int64_t row_off = step + static_cast<int64_t>(j0 + r) * d;
        Kj[r * d + c] = K[row_off + c];
        Vj[r * d + c] = V[row_off + c];
    }
    __syncthreads();

    for (int q0 = 0; q0 < N; q0 += Br) {
        const int br = min(Br, N - q0);

        if (causal && j0 > q0 + br - 1) {
            continue;
        }

        for (int idx = threadIdx.x; idx < br * d; idx += blockDim.x) {
            const int r = idx / d;
            const int c = idx % d;
            const int64_t row_off = step + static_cast<int64_t>(q0 + r) * d;
            Qi[r * d + c] = Q[row_off + c];
            dOi[r * d + c] = dO[row_off + c];
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            float acc = 0.f;
            for (int k = 0; k < d; ++k) {
                acc += __half2float(Qi[r * d + k]) * __half2float(Kj[c * d + k]);
            }
            float s = acc * scale;
            if (causal) {
                const int gq = q0 + r;
                const int gk = j0 + c;
                if (gk > gq) {
                    s = -INFINITY;
                }
            }
            Pbuf[r * bc + c] = s;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            const float Lr = Lse[Lbase + q0 + r];
            const float s = Pbuf[r * bc + c];
            const float p = (s > -1e20f) ? expf(s - Lr) : 0.f;
            Pbuf[r * bc + c] = p;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            float acc = 0.f;
            for (int k = 0; k < d; ++k) {
                acc += __half2float(dOi[r * d + k]) * __half2float(Vj[c * d + k]);
            }
            dPbuf[r * bc + c] = acc;
        }
        __syncthreads();

        for (int r = threadIdx.x; r < br; r += blockDim.x) {
            float sp = 0.f;
            for (int c = 0; c < bc; ++c) {
                sp += Pbuf[r * bc + c] * dPbuf[r * bc + c];
            }
            sum_pdp[r] = sp;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            const float pr = Pbuf[r * bc + c];
            const float dpr = dPbuf[r * bc + c];
            const float dsr = pr * (dpr - sum_pdp[r]);
            Pbuf[r * bc + c] = dsr;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < bc * d; idx += blockDim.x) {
            const int c = idx / d;
            const int kk = idx % d;
            float val = 0.f;
            for (int r = 0; r < br; ++r) {
                val += Pbuf[r * bc + c] * __half2float(Qi[r * d + kk]);
            }
            dKacc[c * d + kk] += scale * val;
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < bc * d; idx += blockDim.x) {
        const int r = idx / d;
        const int kk = idx % d;
        const int64_t out_off = step + static_cast<int64_t>(j0 + r) * d + kk;
        dK[out_off] = __float2half(dKacc[r * d + kk]);
    }
}

// ---------- dV: grid (B, H, num_kv_tiles); sum over all Q tiles i ----------
__global__ void fa2_bwd_dv_kernel(
    half* __restrict__ dV,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ dO,
    const float* __restrict__ Lse,
    int Bc,
    int Br,
    int B,
    int H,
    int N,
    int d,
    int causal)
{
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int bj = blockIdx.z;
    const int j0 = bj * Bc;
    if (j0 >= N) {
        return;
    }
    const int bc = min(Bc, N - j0);
    const int head_stride = N * d;
    const int64_t step = (static_cast<int64_t>(b) * H + h) * head_stride;
    const int64_t Lbase = static_cast<int64_t>(b * H + h) * N;

    extern __shared__ char sh_raw[];
    half* Kj = reinterpret_cast<half*>(sh_raw);
    half* Qi = Kj + Bc * d;
    half* dOi = Qi + Br * d;
    const size_t half_bytes = (static_cast<size_t>(Bc) * d + 2ull * Br * d) * sizeof(half);
    float* Pbuf = reinterpret_cast<float*>(sh_raw + ((half_bytes + 15u) & ~15u));
    float* dVacc = Pbuf + Br * Bc;

    const float scale = rsqrtf(static_cast<float>(d));

    for (int idx = threadIdx.x; idx < Bc * d; idx += blockDim.x) {
        dVacc[idx] = 0.f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < bc * d; idx += blockDim.x) {
        const int r = idx / d;
        const int c = idx % d;
        const int64_t row_off = step + static_cast<int64_t>(j0 + r) * d;
        Kj[r * d + c] = K[row_off + c];
    }
    __syncthreads();

    for (int q0 = 0; q0 < N; q0 += Br) {
        const int br = min(Br, N - q0);

        if (causal && j0 > q0 + br - 1) {
            continue;
        }

        for (int idx = threadIdx.x; idx < br * d; idx += blockDim.x) {
            const int r = idx / d;
            const int c = idx % d;
            const int64_t row_off = step + static_cast<int64_t>(q0 + r) * d;
            Qi[r * d + c] = Q[row_off + c];
            dOi[r * d + c] = dO[row_off + c];
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            float acc = 0.f;
            for (int k = 0; k < d; ++k) {
                acc += __half2float(Qi[r * d + k]) * __half2float(Kj[c * d + k]);
            }
            float s = acc * scale;
            if (causal) {
                const int gq = q0 + r;
                const int gk = j0 + c;
                if (gk > gq) {
                    s = -INFINITY;
                }
            }
            Pbuf[r * bc + c] = s;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < br * bc; idx += blockDim.x) {
            const int r = idx / bc;
            const int c = idx % bc;
            const float Lr = Lse[Lbase + q0 + r];
            const float s = Pbuf[r * bc + c];
            const float p = (s > -1e20f) ? expf(s - Lr) : 0.f;
            Pbuf[r * bc + c] = p;
        }
        __syncthreads();

        for (int idx = threadIdx.x; idx < bc * d; idx += blockDim.x) {
            const int c = idx / d;
            const int kk = idx % d;
            float val = 0.f;
            for (int r = 0; r < br; ++r) {
                val += Pbuf[r * bc + c] * __half2float(dOi[r * d + kk]);
            }
            dVacc[c * d + kk] += val;
        }
        __syncthreads();
    }

    for (int idx = threadIdx.x; idx < bc * d; idx += blockDim.x) {
        const int r = idx / d;
        const int kk = idx % d;
        const int64_t out_off = step + static_cast<int64_t>(j0 + r) * d + kk;
        dV[out_off] = __float2half(dVacc[r * d + kk]);
    }
}

extern "C" void launchMyFA2Backward(
    half* dQ,
    half* dK,
    half* dV,
    const half* Q,
    const half* K,
    const half* V,
    const half* dO,
    const float* Lse,
    int Bc,
    int Br,
    int B,
    int H,
    int N,
    int d,
    int causal)
{
    const dim3 grid_q(
        static_cast<unsigned>(B),
        static_cast<unsigned>(H),
        static_cast<unsigned>(fa2_ceil_div(N, Br)));
    const dim3 grid_kv(
        static_cast<unsigned>(B),
        static_cast<unsigned>(H),
        static_cast<unsigned>(fa2_ceil_div(N, Bc)));
    const unsigned tpb = static_cast<unsigned>(kBwdThreads);

    const size_t half_dq =
        (static_cast<size_t>(Br) * d + 2u * Bc * d + Br * d) * sizeof(half);
    const size_t float_dq =
        (2ull * static_cast<size_t>(Br) * Bc + static_cast<size_t>(Br)) * sizeof(float)
        + static_cast<size_t>(Br) * d * sizeof(float);
    const size_t shmem_dq = ((half_dq + 15u) & ~15u) + float_dq;

    const size_t half_dk =
        (2u * Bc * d + 2u * Br * d) * sizeof(half);
    const size_t float_dk =
        (2ull * static_cast<size_t>(Br) * Bc + static_cast<size_t>(Br)) * sizeof(float)
        + static_cast<size_t>(Bc) * d * sizeof(float);
    const size_t shmem_dk = ((half_dk + 15u) & ~15u) + float_dk;

    const size_t half_dv = (static_cast<size_t>(Bc) * d + 2ull * Br * d) * sizeof(half);
    const size_t float_dv =
        static_cast<size_t>(Br) * Bc * sizeof(float) + static_cast<size_t>(Bc) * d * sizeof(float);
    const size_t shmem_dv = ((half_dv + 15u) & ~15u) + float_dv;

    fa2_bwd_set_max_dynamic_smem(reinterpret_cast<const void*>(fa2_bwd_dq_kernel), shmem_dq);
    fa2_bwd_dq_kernel<<<grid_q, tpb, shmem_dq>>>(dQ, Q, K, V, dO, Lse, Bc, Br, B, H, N, d, causal);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("launchMyFA2Backward dQ CUDA error: %s\n", cudaGetErrorString(err));
    }
    fa2_bwd_set_max_dynamic_smem(reinterpret_cast<const void*>(fa2_bwd_dk_kernel), shmem_dk);
    fa2_bwd_dk_kernel<<<grid_kv, tpb, shmem_dk>>>(dK, Q, K, V, dO, Lse, Bc, Br, B, H, N, d, causal);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("launchMyFA2Backward dK CUDA error: %s\n", cudaGetErrorString(err));
    }
    fa2_bwd_set_max_dynamic_smem(reinterpret_cast<const void*>(fa2_bwd_dv_kernel), shmem_dv);
    fa2_bwd_dv_kernel<<<grid_kv, tpb, shmem_dv>>>(dV, Q, K, dO, Lse, Bc, Br, B, H, N, d, causal);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("launchMyFA2Backward dV CUDA error: %s\n", cudaGetErrorString(err));
    }
}
