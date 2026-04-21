#include <torch/extension.h>
#include <ATen/ATen.h>
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
        size);
    return C;
}

torch::Tensor myFA2(
    torch::Tensor QTensor,
    torch::Tensor KTensor,
    torch::Tensor VTensor,
    torch::Tensor LTensor,
    torch::Tensor MTensor,
    int Bc,
    int Br,
    int B,
    int H,
    int N,
    int d,
    int causal)
{

    auto device = QTensor.device();
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(device);
    auto OTensor = torch::zeros({B, H, N, d}, options);
    TORCH_CHECK(KTensor.device() == device, "KTensor must be on same device as Q");
    TORCH_CHECK(VTensor.device() == device, "VTensor must be on same device as Q");
    TORCH_CHECK(LTensor.device() == device, "LTensor must be on same device as Q");
    TORCH_CHECK(OTensor.device() == device, "O must be on same device as Q");
    TORCH_CHECK(MTensor.device() == device, "M must be on same device as Q");
    TORCH_CHECK(LTensor.dtype() == torch::kFloat32, "LTensor must be float32 (log-sum-exp per query row)");
    TORCH_CHECK(MTensor.dtype() == torch::kFloat32, "MTensor must be float32");

    half* O = (half*)(OTensor.data_ptr<at::Half>());
    half* Q = (half*)(QTensor.data_ptr<at::Half>());
    half* K = (half*)(KTensor.data_ptr<at::Half>());
    half* V = (half*)(VTensor.data_ptr<at::Half>());
    float* l = LTensor.data_ptr<float>();
    float* m = MTensor.data_ptr<float>();
    launchMyFA2(O, Q, K, V, l, m, Bc, Br, B, H, N, d, causal);
    return OTensor;
}

void myFA2_backward(
    torch::Tensor dQTensor,
    torch::Tensor dKTensor,
    torch::Tensor dVTensor,
    torch::Tensor QTensor,
    torch::Tensor KTensor,
    torch::Tensor VTensor,
    torch::Tensor dOTensor,
    torch::Tensor LTensor,
    int Bc,
    int Br,
    int B,
    int H,
    int N,
    int d,
    int causal)
{
    auto device = QTensor.device();
    TORCH_CHECK(dQTensor.device() == device, "dQ must be on same device as Q");
    TORCH_CHECK(LTensor.dtype() == torch::kFloat32, "LTensor must be float32");

    half* dQ = (half*)(dQTensor.data_ptr<at::Half>());
    half* dK = (half*)(dKTensor.data_ptr<at::Half>());
    half* dV = (half*)(dVTensor.data_ptr<at::Half>());
    const half* Q = (const half*)(QTensor.data_ptr<at::Half>());
    const half* K = (const half*)(KTensor.data_ptr<at::Half>());
    const half* V = (const half*)(VTensor.data_ptr<at::Half>());
    const half* dO = (const half*)(dOTensor.data_ptr<at::Half>());
    const float* Lse = LTensor.data_ptr<float>();

    launchMyFA2Backward(dQ, dK, dV, Q, K, V, dO, Lse, Bc, Br, B, H, N, d, causal);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mytest", &mytest, "Test function for matrix addition");
    m.def("myFA2", &myFA2, "FlashAttention-2 style forward (optional causal mask)");
    m.def("myFA2_backward", &myFA2_backward, "FlashAttention-2 style backward (recompute)");
}
