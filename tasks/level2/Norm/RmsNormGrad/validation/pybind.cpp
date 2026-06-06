/**
 * @file extension_rms_norm_grad.cpp
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
std::vector<at::Tensor> rms_norm_grad(at::Tensor dy, at::Tensor x, at::Tensor rstd, at::Tensor gamma)
{
    at::Tensor dxOut = torch::empty_like(x);
    at::Tensor dgammaOut = torch::empty_like(gamma, torch::dtype(torch::kFloat32));

    EXEC_NPU_CMD(aclnnRmsNormGrad, dy, x, rstd, gamma, dxOut, dgammaOut);

    std::vector<at::Tensor> result = {dxOut, dgammaOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("rms_norm_grad", &rms_norm_grad, "");
}