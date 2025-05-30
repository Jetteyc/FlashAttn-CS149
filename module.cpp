// module.cpp
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <iostream>
#include <time.h>
#include <sys/time.h>
#include <vector>
#include <immintrin.h>
#include <cuda_fp16.h>
#include "kernel.h" 
#include <string>
torch::Tensor mytest(torch::Tensor A, torch::Tensor B) {
    auto C = at::empty_like(A);
    int size = A.numel();
    launchMatrixAdd(
        (half*)(A.data_ptr<at::Half>()), 
        (half*)(B.data_ptr<at::Half>()), 
        (half*)(C.data_ptr<at::Half>()), 
        size
    );
    return C;
}


torch::Tensor myFA1(torch::Tensor QTensor, torch::Tensor KTensor, torch::Tensor VTensor, 
    torch::Tensor LTensor, torch::Tensor MTensor, 
    int Bc, int Br,int B, int H, int N, int d) {
        
    auto device = QTensor.device();
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(device);
    auto OTensor = torch::zeros({B, H, N, d}, options);
    TORCH_CHECK(KTensor.device() == device, "KTensor 必须与 QTensor 在同一设备");
    TORCH_CHECK(VTensor.device() == device, "VTensor 必须与 QTensor 在同一设备");
    TORCH_CHECK(LTensor.device() == device, "LTensor 必须与 QTensor 在同一设备");
    TORCH_CHECK(OTensor.device() == device, "O 必须与 QTensor 在同一设备");
    TORCH_CHECK(MTensor.device() == device, "M 必须与 QTensor 在同一设备");
    // Q, K, V are passed in with Shape: (B, H, N, d)
    // Sij, Pij are passed in with Shape: (Br, Bc)
    // Kj, Vj are passed in with Shape: (Bc, d)
    // Qi, Oi, and PV  are passed in with Shape: (Br, d)
    // L, M in passed in with Shape: (N)
    // Li, Lij, and Lnew are passed in with shape (Br)
    // mi, mij, and mnew are passed in with shape (Br)
    half* O = (half*)(OTensor.data_ptr<at::Half>());
    half* Q = (half*)(QTensor.data_ptr<at::Half>());
    half* K = (half*)(KTensor.data_ptr<at::Half>());
    half* V = (half*)(VTensor.data_ptr<at::Half>());
    half* l = (half*)(LTensor.data_ptr<at::Half>());
    half* m = (half*)(MTensor.data_ptr<at::Half>());
    launchMyFA1(O, Q, K, V, l, m, Bc, Br, B, H, N, d);
    // cudaDeviceReset();
    cudaDeviceSynchronize();
    return torch::from_blob(O, {B, H, N, d}, torch::TensorOptions().dtype(torch::kHalf).device(device));
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mytest", &mytest, "Test function for matrix addition");
    m.def("myFA1", &myFA1, "my Flash Attention - 1");
}