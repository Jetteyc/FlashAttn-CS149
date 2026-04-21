#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_fp16.h>

extern "C" void launchMatrixAdd(half* A, half* B, half* C, int size);
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
    int causal);
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
    int causal);
#endif // KERNEL_H
