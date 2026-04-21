// Minimal CUDA helpers (matrix add smoke test). FlashAttention-2 lives in kernel_fa2.cu.

#include <stdexcept>
#include <string>
#include "kernel.h"
#include <stdio.h>
#include <cuda_fp16.h>

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
