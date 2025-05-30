// kernel.h
#ifndef KERNEL_H
#define KERNEL_H

// #include <cuda_fp16.h>

extern "C" void launchMatrixAdd(float* A, float* B, float* C, int size);
extern "C" void launchMyFA1(float* O, float* Q, float* K, float* V, float* l, float* m, int Bc, int Br,int B, int H, int N, int d);
#endif // KERNEL_H