/**
 * @file extension_deep_norm_grad.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * Implementation note.
 * Implementation note.
 * Implementation note.
 *
 * Implementation note.
 * Implementation note.
 * Implementation note.
 *
 * Implementation note.
 */
std::vector<at::Tensor> deep_norm_grad(at::Tensor dy, at::Tensor x, at::Tensor gx, at::Tensor gamma, at::Tensor mean, at::Tensor rstd, double alpha)
{
    // Output tensors: dxOut, dgxOut, dbetaOut, dgammaOut
    at::Tensor dxOut = torch::empty_like(x);
    at::Tensor dgxOut = torch::empty_like(gx);
    at::Tensor dbetaOut = torch::empty_like(gamma, torch::dtype(torch::kFloat32)); // dbeta and dgamma have same shape as beta/gamma
    at::Tensor dgammaOut = torch::empty_like(gamma, torch::dtype(torch::kFloat32));

    EXEC_NPU_CMD(aclnnDeepNormGrad, dy, x, gx, gamma, mean, rstd, alpha, dxOut, dgxOut, dbetaOut, dgammaOut);

    std::vector<at::Tensor> result = {dxOut, dgxOut, dbetaOut, dgammaOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `deep_norm_grad`
    m.def("deep_norm_grad", &deep_norm_grad, "");
}