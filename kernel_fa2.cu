// FlashAttention-2 forward — structure aligned with Tri Dao / blog (rossiXYZ):
//
// (1) Fewer non-matmul / rescale FLOPs: inner KV loop only does
//     Oacc <- exp(m_old - m_new) * Oacc + (P @ V) with P = exp(S - m_new) unnormalized;
//     one division per row at the very end: O = Oacc / l (no per-tile full row normalize).
//
// (2) Sequence-parallel: grid (B, H, ceil(N/Br)); each block owns one Q tile.
//     Memory order: load entire Qi tile first, __syncthreads(); then inner for KV:
//     load Kj, Vj, compute, update state.
//
// (3) split-Q across warps (NUM_WARPS=4): row r is handled by warp (r % NUM_WARPS).
//     That warp's 32 lanes jointly reduce max / sum over columns (shuffle), compute P,
//     and accumulate Oacc[r,*] without cross-warp softmax communication on those rows.

#include <cuda_fp16.h>
#include <cstdint>
#include <math.h>
#include <stdio.h>
#include "kernel.h"

constexpr int kNumWarps = 4;
constexpr int kWarpThreads = 32;
constexpr int kBlockThreads = kNumWarps * kWarpThreads;

__device__ __forceinline__ float warp_reduce_max(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, mask));
    }
    return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    }
    return v;
}

__global__ void __launch_bounds__(kBlockThreads, 1) flash_attention_fwd_fa2_kernel(
    half* __restrict__ O,
    const half* __restrict__ Q,
    const half* __restrict__ K,
    const half* __restrict__ V,
    float* __restrict__ l_out,
    float* __restrict__ m_out,
    int Bc,
    int Br,
    int B,
    int H,
    int N,
    int d,
    int causal)
{
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int bid_z = blockIdx.z;
    const int q0 = bid_z * Br;
    if (q0 >= N) {
        return;
    }
    const int br = min(Br, N - q0);
    const int head_stride = N * d;
    const int64_t step = (static_cast<int64_t>(b) * H + h) * head_stride;

    extern __shared__ char sh_raw[];
    half* Qi = reinterpret_cast<half*>(sh_raw);
    half* Kj = Qi + Br * d;
    half* Vj = Kj + Bc * d;
    float* S = reinterpret_cast<float*>(Vj + Bc * d);
    float* Oacc = S + Br * Bc;
    float* m_row = Oacc + Br * d;
    float* l_row = m_row + Br;

    const float scale = rsqrtf(static_cast<float>(d));

    // --- (2) Load Q tile first (all warps cooperate) ---
    for (int idx = threadIdx.x; idx < br * d; idx += blockDim.x) {
        const int r = idx / d;
        const int c = idx % d;
        Qi[r * d + c] = Q[step + (q0 + r) * d + c];
    }
    __syncthreads();

    for (int r = threadIdx.x; r < br; r += blockDim.x) {
        m_row[r] = -INFINITY;
        l_row[r] = 0.f;
    }
    __syncthreads();

    for (int idx = threadIdx.x; idx < br * d; idx += blockDim.x) {
        Oacc[idx] = 0.f;
    }
    __syncthreads();

    // --- Inner loop: KV tiles (load K,V after Q is resident) ---
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

        // (3) GEMM S = scale * Q K^T : split rows across warps (split-Q)
        for (int r = warp_id; r < br; r += kNumWarps) {
            for (int c = lane_id; c < bc; c += kWarpThreads) {
                float acc = 0.f;
                for (int k = 0; k < d; ++k) {
                    acc += __half2float(Qi[r * d + k]) * __half2float(Kj[c * d + k]);
                }
                S[r * bc + c] = acc * scale;
            }
        }
        __syncthreads();

        if (causal) {
            for (int r = warp_id; r < br; r += kNumWarps) {
                for (int c = lane_id; c < bc; c += kWarpThreads) {
                    const int gq = q0 + r;
                    const int gk = j + c;
                    if (gk > gq) {
                        S[r * bc + c] = -INFINITY;
                    }
                }
            }
            __syncthreads();
        }

        // (3) Online softmax + Oacc update: row r owned by warp (r % kNumWarps)
        for (int r = warp_id; r < br; r += kNumWarps) {
            const float m_old = m_row[r];

            float lane_max = -INFINITY;
            for (int c = lane_id; c < bc; c += kWarpThreads) {
                lane_max = fmaxf(lane_max, S[r * bc + c]);
            }
            const float m_new = warp_reduce_max(lane_max);

            float lane_sum = 0.f;
            for (int c = lane_id; c < bc; c += kWarpThreads) {
                const float p = expf(S[r * bc + c] - m_new);
                S[r * bc + c] = p;
                lane_sum += p;
            }
            const float l_ij = warp_reduce_sum(lane_sum);

            const float alpha = expf(m_old - m_new);
            const float l_new = alpha * l_row[r] + l_ij;

            for (int k = lane_id; k < d; k += kWarpThreads) {
                float val = 0.f;
                for (int c = 0; c < bc; ++c) {
                    val += S[r * bc + c] * __half2float(Vj[c * d + k]);
                }
                Oacc[r * d + k] = alpha * Oacc[r * d + k] + val;
            }

            if (lane_id == 0) {
                m_row[r] = m_new;
                l_row[r] = l_new;
            }
            __syncwarp();
        }
        __syncthreads();
    }

    // (1) Single rescale per row after all KV tiles
    for (int idx = threadIdx.x; idx < br * d; idx += blockDim.x) {
        const int r = idx / d;
        const int k = idx % d;
        const float inv_l = 1.f / fmaxf(l_row[r], 1e-20f);
        const int64_t out_off = step + static_cast<int64_t>(q0 + r) * d + k;
        O[out_off] = __float2half(Oacc[r * d + k] * inv_l);
    }
    __syncthreads();

    if (l_out != nullptr && m_out != nullptr) {
        for (int r = threadIdx.x; r < br; r += blockDim.x) {
            const int64_t lm_off = static_cast<int64_t>(b * H + h) * N + (q0 + r);
            const float L_stat = m_row[r] + logf(fmaxf(l_row[r], 1e-20f));
            l_out[lm_off] = L_stat;
            m_out[lm_off] = m_row[r];
        }
    }
}

extern "C" void launchMyFA2(
    half* O,
    half* Q,
    half* K,
    half* V,
    float* l,
    float* m,
    int Bc,
    int Br,
    int B,
    int H,
    int N,
    int d,
    int causal)
{
    const size_t shmem_bytes =
        (static_cast<size_t>(2) * Br * d + static_cast<size_t>(2) * Bc * d) * sizeof(half)
        + (static_cast<size_t>(Br) * Bc + static_cast<size_t>(Br) * d + static_cast<size_t>(2) * Br) * sizeof(float);

    const dim3 grid(static_cast<unsigned>(B), static_cast<unsigned>(H),
                    static_cast<unsigned>((N + Br - 1) / Br));
    const unsigned tpb = static_cast<unsigned>(kBlockThreads);

    flash_attention_fwd_fa2_kernel<<<grid, tpb, shmem_bytes>>>(
        O, Q, K, V, l, m, Bc, Br, B, H, N, d, causal);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("launchMyFA2 CUDA error: %s\n", cudaGetErrorString(err));
    }
}
