// kernel.h
#ifndef KERNEL_H
#define KERNEL_H

#include <cuda_fp16.h>

extern "C" void launchMatrixAdd(half* A, half* B, half* C, int size);
extern "C" void launchMyFA1(
    half* O, half* Q, half* K, half* V, half* l, half* m,
    int Bc, int Br, int B, int H, int N, int d, int use_wmma
);
#endif // KERNEL_H