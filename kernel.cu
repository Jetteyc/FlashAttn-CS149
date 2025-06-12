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
// #define ENABLE_CHECK
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


__device__ void check(const half* A, int N, int M, const char* label) {
#ifdef ENABLE_CHECK
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0) {
        printf("=== check: %s ===\n", label);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                float val = __half2float(A[i * M + j]);
                printf("%.8f ", val);
            }
            printf("\n");
        }
        printf("==================\n");
    }
#endif
}
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
    half* lij, const half* li,
    half* mij, const half* mi,
    int d
) {
    int j = threadIdx.x;
    if(j < br) {
        lij[j] = CUDART_ZERO_FP16;
        mij[j] = mi[j];
    }
    __syncthreads();
    if (j < bc) {
        for (int i = 0; i < br; i++) {
            Sij[i * bc + j] = CUDART_ZERO_FP16;
            for (int k = 0; k < d; k++) {
                Sij[i * bc + j] = __hadd(Sij[i * bc + j] , __hmul(Qi[i * d + k], Kj[j * d + k]));
            }
        }
    }
    // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> C_frag;
    // wmma::fill_fragment(C_frag, CUDART_ZERO_FP16);
    // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> Qi_frag;
    // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> Kj_frag;

    // wmma::load_matrix_sync(Qi_frag, Qi, d);
    // wmma::load_matrix_sync(Kj_frag, Kj, bc);

    // wmma::mma_sync(C_frag, Qi_frag, Kj_frag, C_frag);
    // wmma::store_matrix_sync(Sij, C_frag, bc, wmma::mem_row_major);

    
    __syncthreads();
    if (j < br) {
        for (int i = 0; i < bc; i++) {
            Sij[j * bc + i] = __hdiv(Sij[j * bc + i], __float2half(sqrt(d)));
            mij[j] = __hmax(mij[j], Sij[j * bc + i]);
        }
    }
    __syncthreads();
    if (j < br) {
        for (int i = 0; i < bc; i++) {
            half exp_val = hexp(__hsub(Sij[j * bc + i], mij[j]));
            Pij[j * bc + i] = exp_val;
            lij[j] = __hadd(lij[j], exp_val);
        }
        lij[j] = __hadd(lij[j], __hmul(li[j], hexp(__hsub(mi[j], mij[j]))));
    }
    __syncthreads();
}


__device__ void updateOutput(
    const half* Pij, int br, int bc, const half* Vj, int d,
    half* Oi, const half* li,
    half* mij, half* mi
) {
    int idx = threadIdx.x;
    int stride = blockDim.x;
    for (int ik = idx; ik < br * d; ik += stride) {
        int i = ik / d;
        int k = ik % d;
        half val = CUDART_ZERO_FP16;
        for (int j = 0; j < bc; j++) {
            val =__hadd(val, __hmul(Pij[i * bc + j], Vj[j * d + k]));
        }
        Oi[ik] = __hadd(__hmul(hexp(__hsub(mi[i], mij[i])),Oi[ik]), val);
    }
    __syncthreads();
}

__global__ void myFA1Kernel(
    half* O, half* Q, half* K, half* V, half* l, half* m, 
    int Bc, int Br,int B, int H, int N, int d
){
    int b = blockIdx.x; 
    int h = blockIdx.y;
    int i = blockIdx.z * Br;
    int tx = threadIdx.x;
    int step = b * H * N * d + h * N * d;
    int lm_offset = b * H * N + h * N;
    extern __shared__ half shared_mem[];
    half* Qi = shared_mem; // (Br, d)
    half* Kj = Qi + Br * d; // (Bc, d)
    half* Vj = Kj + Bc * d; // (Bc, d)
    half* Oi = Vj + Bc * d; // (Br, d)
    half* Sij = Oi + Br * d; // (Br, Bc)
    half* Pij = Sij + Br * Bc; // (Br, Bc)
    half* li = Pij + Br * Bc; // (Br)
    half* mi = li + Br; // (Br)
    half* lij = mi + Br; // (Br)
    half* mij = lij + Br; // (Br)
    
    check(K, N, d, "K");
    check(V, N, d, "V");
    check(Q, N, d, "Q");
    // load Qi, Oi, li
    loadMatrix(Q + step + i * d, Qi, min(N, i + Br) - i, d);
    loadMatrix(O + step + i * d, Oi, min(N, i + Br) - i, d);
    if(tx < min(N, i + Br) - i){
        li[tx] = l[lm_offset + i + tx];
        mi[tx] = m[lm_offset + i + tx];
    }
    check(Qi, min(N, i + Br) - i, d, "Qi");
    check(Oi, min(N, i + Br) - i, d, "Oi");
    check(li, 1, min(N, i + Br) - i, "li");
    check(mi, 1, min(N, i + Br) - i, "mi");
    __syncthreads();
    for (int j = 0; j < N; j += Bc) {
        // load Kj, Vj
        loadMatrix(K + step + j * d, Kj, min(N, j + Bc) - j, d);
        loadMatrix(V + step + j * d, Vj, min(N, j + Bc) - j, d);
        check(Kj, min(N, j + Bc) - j, d, "Kj");
        check(Vj, min(N, j + Bc) - j, d, "Vj");
        
        // Sij = QiKj_t/sqrtf(d), Pij = exp(Sij), Lij = rowsum(Pij), Lnew = Li + Lij
        computeAttention(Qi, min(N, i + Br) - i, Kj, min(N, j + Bc) - j, Sij, Pij, lij, li,mij, mi, d);
        check(Sij, min(N, i + Br) - i, min(N, j + Bc) - j, "Sij");
        check(Pij, min(N, i + Br) - i, min(N, j + Bc) - j, "Pij");
        check(lij, 1, min(N, i + Br) - i, "lij");
        check(mij, 1, min(N, i + Br) - i, "mij");
        // Oi <- (liOi + PijVj) / lnew
        check(Oi, min(N, i + Br) - i, d, "Oi - before");
        updateOutput(Pij, min(N, i + Br) - i, min(N, j + Bc) - j, Vj, d, Oi, li, mij, mi);
        check(Oi, min(N, i + Br) - i, d, "Oi - after");
        // Write Oi, lnew to O and L;
        if(tx < min(N, i + Br) - i) {
            li[tx] = lij[tx];
            mi[tx] = mij[tx];
        }
    }
    __syncthreads();
    if(tx < min(N, i + Br) - i) {
        for (int j = 0; j < d; j++) {
            Oi[tx * d + j] = __hdiv(Oi[tx * d + j], li[tx]);
        }
        l[lm_offset + i + tx] = li[tx];
        m[lm_offset + i + tx] = mi[tx];
    }
    __syncthreads();
    loadMatrix(Oi, O + step + i * d, min(N, i + Br) - i, d);
    
}
extern "C" void launchMyFA1(half* O, half* Q, half* K, half* V, half* l, half* m, int Bc, int Br,int B, int H, int N, int d
){
    const int sram_size = (2 * Br * d + 2 * Bc * d + 2 * Br * Bc + 4 * Br) * sizeof(half);
    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    if(Br > Bc) {
        printf("Br > Bc\n");
        return;
    }
    printf("B = %d, H = %d, N = %d, d = %d\nBr = %d, Bc = %d\nMax shared memory: %d, requested shared memory: %d \n\n", B, H, N, d, Br, Bc, max_sram_size, sram_size);
    dim3 blocks(B, H, (N + Br - 1) / Br);
    dim3 threads(Bc);
    myFA1Kernel<<<blocks, threads, sram_size>>>(O, Q, K, V, l, m, Bc, Br, B, H, N, d);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
}