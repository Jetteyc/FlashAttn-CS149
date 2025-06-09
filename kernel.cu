// kernel.cu
#include <stdexcept>
#include <string>
#include "kernel.h"
#include <stdio.h>
#include <float.h>
#include <mma.h> 
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_SIZE 32
#define CUDART_NEG_INF_FP16 __ushort_as_half(0xFC00); // IEEE 754 半精度负无穷的十六进制表示
using namespace nvcuda; 
__global__ void matrixAddKernel(half* A, half* B, half* C, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        C[i] = __hadd(A[i], B[i]);
    }
}

extern "C" void launchMatrixAdd(half* A, half* B, half* C, int size) {
    int threads_per_block = 256;
    int blocks_per_grid = (size + threads_per_block - 1) / threads_per_block;

    matrixAddKernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, size);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA 错误: ") + cudaGetErrorString(err));
    }
}


// __device__ void check(const half* A, int N, int M, const char* label) {
// #ifdef ENABLE_CHECK
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;
//     if (tid == 0) {
//         printf("=== check: %s ===\n", label);
//         for (int i = 0; i < N; ++i) {
//             for (int j = 0; j < M; ++j) {
//                 float val = __half2float(A[i * M + j]);
//                 printf("%.8f ", val);
//             }
//             printf("\n");
//         }
//         printf("==================\n");
//     }
// #endif
// }

__device__ void loadMatrix(const half* M, half* N, int tile_size, int d) {
    // printf("loadMatrix called by thread %d\n", threadIdx.x);
    int i = threadIdx.x;
    if (i < tile_size) {
        for(int j = 0; j < d; j++) {
            N[i * d + j] = M[i * d + j];
        }
    }
    __syncthreads();
}

__device__ void computeAttention(
    const half* Qi, int br, const half* Kj, int bc,
    half* Sij, half* Pij, 
    half* lij, const half* li, half* lnew, 
    half* mij, const half* mi, half* mnew,
    int d
) {
    int j = threadIdx.x;
    if (j < bc) {
        for (int i = 0; i < br; i++) {
            Sij[i * bc + j] = CUDART_ZERO_FP16;
        }
    }
    if(j < br) {
        lij[j] = CUDART_ZERO_FP16;
        mij[j] = CUDART_NEG_INF_FP16;
    }
    __syncthreads();
    // if (j < bc) {
    //     for (int i = 0; i < br; i++) {
    //         for (int k = 0; k < d; k++) {
    //             Sij[i * bc + j] = __hadd(Sij[i * bc + j] , __hmul(Qi[i * d + k], Kj[j * d + k]));
    //         }
    //     }
    // }
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    wmma::fill_fragment(C_frag, CUDART_ZERO_FP16);
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Qi_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> Kj_frag;

    wmma::load_matrix_sync(Qi_frag, Qi, d);
    wmma::load_matrix_sync(Kj_frag, Kj, bc);

    wmma::mma_sync(C_frag, Qi_frag, Kj_frag, C_frag);
    wmma::store_matrix_sync(Sij, C_frag, bc, wmma::mem_row_major);

    __syncthreads();
    // check(Sij, br, bc, "Sij - 1");
    if (j < bc) {
        for (int i = 0; i < br; i++) {
            half val = Sij[i * bc + j];
            Sij[i * bc + j] = __hdiv(val, __float2half(sqrt(d)));
        }
    }
    __syncthreads();
    if (j < br) {
        for (int i = 0; i < bc; i++) {
            mij[j] = __hmax(mij[j], Sij[j * bc + i]);
#ifdef ENABLE_CHECK
            // printf("got %f, m[%d] = %f\n", val, i, mij[i]);
#endif
        }
    }
    __syncthreads();
    if (j < bc) {
        for (int i = 0; i < br; i++) {
            half exp_val = hexp(__hsub(Sij[i * bc + j], mij[i]));
            Pij[i * bc + j] = exp_val;
            atomicAdd(&lij[i], exp_val);
        }
    }
    __syncthreads();
    if(j < br){
        mnew[j] = __hmax(mi[j], mij[j]);
        lnew[j] = __hadd(__hmul(hexp(__hsub(mi[j], mnew[j])), li[j]), __hmul(hexp(__hsub(mij[j], mnew[j])), lij[j]));
    }
    __syncthreads();
}


__device__ void updateOutput(
    const half* Pij, int Br, int Bc, const half* Vj, int d,
    half* Oi, half* lnew, const half* li,
    half* mij, half* mi, half* mnew
) {
    int idx = threadIdx.x;
    int stride = blockDim.x;
    for (int ik = idx; ik < Br * d; ik += stride) {
        int i = ik / d;
        int k = ik % d;
        half val = CUDART_ZERO_FP16;
        for (int j = 0; j < Bc; j++) {
            val =__hadd(val, __hmul(Pij[i * Bc + j], Vj[j * d + k]));
        }
        Oi[ik] = __hdiv(__hadd(__hmul(__hmul(hexp(__hsub(mi[i], mnew[i])),Oi[ik]),li[i]),__hmul(val, hexp(__hsub(mij[i], mnew[i])))),lnew[i]);
    }
    __syncthreads();
}

__global__ void myFA1Kernel(
    half* O, half* Q, half* K, half* V, half* l, half* m, 
    int Bc, int Br,int B, int H, int N, int d
){
    int b = blockIdx.x; 
    int h = blockIdx.y;
    int tx = threadIdx.x;
    int step = b * H * N * d + h * N * d;
    int lm_offset = b * H * N + h * N;
    // LTensor.zero_();
    extern __shared__ half shared_mem[];
    half* Qi = shared_mem; // (Br, d)
    half* Kj = Qi + Br * d; // (Bc, d)
    half* Vj = Kj + Bc * d; // (Bc, d)
    half* Oi = Vj + Bc * d; // (Br, d)
    half* Sij = Oi + Br * d; // (Br, Bc)
    half* Pij = Sij + Br * Bc; // (Br, Bc)
    half* li = Pij + Br * Bc; // (Br)
    half* lnew = li + Br; // (Br)
    half* mi = lnew + Br; // (Br)
    half* mnew = mi + Br; // (Br)
    half* lij = mnew + Br; // (Br)
    half* mij = lij + Br; // (Br)
    
    // check(K, N, d, "K");
    // check(V, N, d, "V");
    // check(Q, N, d, "Q");
    for (int i = 0; i < N; i += Br) {
        // load Qi, Oi, li
        loadMatrix(Q + step + i * d, Qi, min(N, i + Br) - i, d);
        loadMatrix(O + step + i * d, Oi, min(N, i + Br) - i, d);
        if(tx < Br){
            lnew[tx] = CUDART_ZERO_FP16;
            mnew[tx] = CUDART_NEG_INF_FP16;
            li[tx] = l[lm_offset + i + tx];
            mi[tx] = m[lm_offset + i + tx];
        }
        // check(Qi, min(N, i + Br) - i, d, "Qi");
        // check(Oi, min(N, i + Br) - i, d, "Oi");
        // check(li, 1, min(N, i + Br) - i, "li");
        // check(mi, 1, min(N, i + Br) - i, "mi");
        // check(lnew, 1, min(N, i + Br) - i, "lnew-before");
        // check(mnew, 1, min(N, i + Br) - i, "mnew-before");
        __syncthreads();
        for (int j = 0; j < N; j += Bc) {
            // load Kj, Vj
            loadMatrix(K + step + j * d, Kj, min(N, j + Bc) - j, d);
            loadMatrix(V + step + j * d, Vj, min(N, j + Bc) - j, d);
            // check(Kj, min(N, j + Bc) - j, d, "Kj");
            // check(Vj, min(N, j + Bc) - j, d, "Vj");
            
            // Sij = QiKj_t/sqrtf(d), Pij = exp(Sij), Lij = rowsum(Pij), Lnew = Li + Lij
            computeAttention(Qi, min(N, i + Br) - i, Kj, min(N, j + Bc) - j, Sij, Pij, lij, li, lnew, mij, mi, mnew, d);
            // check(Sij, min(N, i + Br) - i, min(N, j + Bc) - j, "Sij");
            // check(Pij, min(N, i + Br) - i, min(N, j + Bc) - j, "Pij");
            // check(lij, 1, min(N, i + Br) - i, "lij");
            // check(li, 1, min(N, i + Br) - i, "li");
            // check(lnew, 1, min(N, i + Br) - i, "lnew");
            // check(mij, 1, min(N, i + Br) - i, "mij");
            // check(mi, 1, min(N, i + Br) - i, "mi");
            // check(mnew, 1, min(N, i + Br) - i, "mnew");
            // Oi <- (liOi + PijVj) / lnew
            updateOutput(Pij, min(N, i + Br) - i, min(N, j + Bc) - j, Vj, d, Oi, lnew, li, mij, mi, mnew);
            // Write Oi, lnew to O and L;
            __syncthreads();
        }
        loadMatrix(Oi, O + step + i * d, min(N, i + Br) - i, d);
        if(tx < min(N, i + Br) - i) {
            l[lm_offset + i + tx] = lnew[tx];
            m[lm_offset + i + tx] = mnew[tx];
        }
        __syncthreads();
    }
}
extern "C" void launchMyFA1(half* O, half* Q, half* K, half* V, half* l, half* m, int Bc, int Br,int B, int H, int N, int d
){
    Bc = 16;
    Br = 16;
    const int sram_size = (2 * Br * d + 2 * Bc * d + 2 * Br * Bc + 6 * Br) * sizeof(half);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if(Br > Bc) {
        printf("Br > Bc\n");
        return;
    }
    printf("\nBr = %d, Bc = %d\nMax shared memory: %d, requested shared memory: %d \n\n", Br, Bc, max_sram_size, sram_size);
    dim3 blocks(B, H);
    dim3 threads(WARP_SIZE);
    myFA1Kernel<<<blocks, threads, sram_size>>>(O, Q, K, V, l, m, Bc, Br, B, H, N, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
}