// kernel.cu
#include <stdexcept>
#include <string>
#include "kernel.h"
#include <stdio.h>
#include <float.h>

__global__ void matrixAddKernel(float* A, float* B, float* C, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

extern "C" void launchMatrixAdd(float* A, float* B, float* C, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    matrixAddKernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA 错误: ") + cudaGetErrorString(err));
    }
}


__device__ __noinline__ void check(const float* A, int N, int M, const char* label) {
#ifdef ENABLE_CHECK
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (tid == 0) {
        printf("=== check: %s ===\n", label);
        for (int i = 0; i < N; ++i) {
            if(i < 10 || i >= N - 10){
                for (int j = 0; j < M; ++j) {
                    float val = A[i * M + j];
                    printf("%.8f ", val);
                }
                printf("\n");
            }   
            if(i == 10){
                printf("...\n");
            }
        }
        printf("==================\n");
    }
#endif
}

__device__ __forceinline__ float atomicMax(float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}
__device__ __noinline__ void loadMatrix(const float* M, float* N, int tile_size, int d) {
    // printf("loadMatrix called by thread %d\n", threadIdx.x);
    int i = threadIdx.x;
    if (i < tile_size) {
        for(int j = 0; j < d; j++) {
            N[i * d + j] = M[i * d + j];
        }
    }
    __syncthreads();
}


__device__ __noinline__ void computeAttention(
    const float* Qi, int br, const float* Kj, int bc,
    float* Sij, float* Pij, 
    float* lij, const float* li, float* lnew, 
    float* mij, const float* mi, float* mnew,
    int d
) {
    int j = threadIdx.x;
    if (j < bc) {
        for (int i = 0; i < br; i++) {
            Sij[i * bc + j] = 0.0f;
        }
    }
    if(j < br) {
        lij[j] = 0;
        mij[j] = -FLT_MAX;
    }
    __syncthreads();
    if (j < bc) {
        for (int i = 0; i < br; i++) {
            for (int k = 0; k < d; k++) {
                Sij[i * bc + j] = Sij[i * bc + j] + Qi[i * d + k] * Kj[j * d + k];
            }
        }
    }
    __syncthreads();
    // check(Sij, br, bc, "Sij - 1");
    if (j < bc) {
        for (int i = 0; i < br; i++) {
            float val = Sij[i * bc + j];
            Sij[i * bc + j] = val / sqrtf(d);
        }
    }
    __syncthreads();
    if (j < br) {
        for (int i = 0; i < bc; i++) {
            mij[j] = max(mij[j], Sij[j * bc + i]);
#ifdef ENABLE_CHECK
            // printf("got %f, m[%d] = %f\n", val, i, mij[i]);
#endif
        }
    }
    __syncthreads();
    if (j < bc) {
        for (int i = 0; i < br; i++) {
            float exp_val = expf(Sij[i * bc + j] - mij[i]);
            Pij[i * bc + j] = exp_val;
            atomicAdd(&lij[i], exp_val);
        }
    }
    __syncthreads();
    if(j < br){
        mnew[j] = max(mi[j], mij[j]);
        lnew[j] = exp(mi[j] - mnew[j]) * li[j] + exp(mij[j] - mnew[j]) * lij[j];
    }
    __syncthreads();
}


__device__ __noinline__ void updateOutput(
    const float* Pij, int Br, int Bc, const float* Vj, int d,
    float* Oi, float* lnew, const float* li,
    float* mij, float* mi, float* mnew
) {
    int idx = threadIdx.x;
    int stride = blockDim.x;
    for (int ik = idx; ik < Br * d; ik += stride) {
        int i = ik / d;
        int k = ik % d;
        float val = 0;
        for (int j = 0; j < Bc; j++) {
            val += Pij[i * Bc + j] * Vj[j * d + k];
        }
        Oi[ik] = (exp(mi[i] - mnew[i]) * Oi[ik] * li[i] + val * exp(mij[i] - mnew[i])) / lnew[i];
    }
    __syncthreads();
}

__global__ void myFA1Kernel(
    float* O, float* Q, float* K, float* V, float* l, float* m, 
    int Bc, int Br,int B, int H, int N, int d
){
    int b = blockIdx.x; 
    int h = blockIdx.y;
    int tx = threadIdx.x;
    int step = b * H * N * d + h * N * d;
    int lm_offset = b * H * N + h * N;
    // LTensor.zero_();
    extern __shared__ float shared_mem[];
    float* Qi = shared_mem; // (Br, d)
    float* Kj = Qi + Br * d; // (Bc, d)
    float* Vj = Kj + Bc * d; // (Bc, d)
    float* Oi = Vj + Bc * d; // (Br, d)
    float* Sij = Oi + Br * d; // (Br, Bc)
    float* Pij = Sij + Br * Bc; // (Br, Bc)
    float* li = Pij + Br * Bc; // (Br)
    float* lnew = li + Br; // (Br)
    float* mi = lnew + Br; // (Br)
    float* mnew = mi + Br; // (Br)
    float* lij = reinterpret_cast<float*>(mnew) + Br; // (Br)
    float* mij = lij + Br; // (Br)
    
    check(K, N, d, "K");
    check(V, N, d, "V");
    check(Q, N, d, "Q");
    for (int j = 0; j < N; j += Bc) {
        // load Kj, Vj
        loadMatrix(K + step + j * d, Kj, min(N, j + Bc) - j, d);
        loadMatrix(V + step + j * d, Vj, min(N, j + Bc) - j, d);
        check(Kj, min(N, j + Bc) - j, d, "Kj");
        check(Vj, min(N, j + Bc) - j, d, "Vj");
        for (int i = 0; i < N; i += Br) {
            if(tx < Br){
                lnew[tx] = 0.0f;
                mnew[tx] = -FLT_MAX;
                li[tx] = l[lm_offset + i + tx];
                mi[tx] = m[lm_offset + i + tx];
            }
            __syncthreads();
            // load Qi, Oi, li
            loadMatrix(Q + step + i * d, Qi, min(N, i + Br) - i, d);
            check(Qi, min(N, i + Br) - i, d, "Qi");
            loadMatrix(O + step + i * d, Oi, min(N, i + Br) - i, d);
            check(Oi, min(N, i + Br) - i, d, "Oi");
            check(li, 1, min(N, i + Br) - i, "li");
            check(mi, 1, min(N, i + Br) - i, "mi");
            // Sij = QiKj_t/sqrtf(d), Pij = exp(Sij), Lij = rowsum(Pij), Lnew = Li + Lij
            check(lnew, 1, min(N, i + Br) - i, "lnew-before");
            check(mnew, 1, min(N, i + Br) - i, "mnew-before");
            computeAttention(Qi, min(N, i + Br) - i, Kj, min(N, j + Bc) - j, Sij, Pij, lij, li, lnew, mij, mi, mnew, d);
            check(Sij, min(N, i + Br) - i, min(N, j + Bc) - j, "Sij");
            check(Pij, min(N, i + Br) - i, min(N, j + Bc) - j, "Pij");
            check(lij, 1, min(N, i + Br) - i, "lij");
            check(li, 1, min(N, i + Br) - i, "li");
            check(lnew, 1, min(N, i + Br) - i, "lnew");
            check(mij, 1, min(N, i + Br) - i, "mij");
            check(mi, 1, min(N, i + Br) - i, "mi");
            check(mnew, 1, min(N, i + Br) - i, "mnew");
            // Oi <- (liOi + PijVj) / lnew
            updateOutput(Pij, min(N, i + Br) - i, min(N, j + Bc) - j, Vj, d, Oi, lnew, li, mij, mi, mnew);
            // Write Oi, lnew to O and L;
            loadMatrix(Oi, O + step + i * d, min(N, i + Br) - i, d);
            if(tx < min(N, i + Br) - i) {
                l[lm_offset + i + tx] = lnew[tx];
                m[lm_offset + i + tx] = mnew[tx];
            }
            __syncthreads();
        }
    }
    
}
extern "C" void launchMyFA1(float* O, float* Q, float* K, float* V, float* l, float* m, int Bc, int Br,int B, int H, int N, int d
){
    const int sram_size = (2 * Br * d + 2 * Bc * d + 2 * Br * Bc + 4 * Br) * sizeof(float) + 2 * Br * sizeof(float);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if(Br > Bc) {
        printf("Br > Bc\n");
        return;
    }
    printf("\nBr = %d, Bc = %d\nMax shared memory: %d, requested shared memory: %d \n\n", Br, Bc, max_sram_size, sram_size);
    dim3 blocks(B, H);
    dim3 threads(Bc);
    myFA1Kernel<<<blocks, threads, sram_size>>>(O, Q, K, V, l, m, Bc, Br, B, H, N, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}