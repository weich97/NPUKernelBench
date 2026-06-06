/**
 * @file extension_add_layer_norm_grad.cpp
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
std::vector<at::Tensor> add_layer_norm_grad(
    at::Tensor dy,
    at::Tensor x1,
    at::Tensor x2,
    at::Tensor rstd,
    at::Tensor mean,
    at::Tensor gamma,
    c10::optional<at::Tensor> dsumOptional)
{
    // Output tensors: dxOut, dgammaOut, dbetaOut
    // dxOut has the same shape and dtype as x1 (or x2, dy)
    at::Tensor dxOut = torch::empty_like(dy);

    // dgammaOut and dbetaOut have shape of normalized_shape and dtype of gamma/beta respectively
    // We need to infer the shape of dgammaOut and dbetaOut based on gamma.
    // The input `gamma` directly provides its shape.
    at::Tensor dgammaOut = torch::empty_like(gamma, torch::dtype(torch::kFloat32));
    at::Tensor dbetaOut = torch::empty_like(gamma, torch::dtype(torch::kFloat32)); // beta has same shape as gamma

    // Prepare optional dsum tensor for NPU_CMD
    at::Tensor dsum_npu = dsumOptional.has_value() ? dsumOptional.value() : at::Tensor();

    // Call the NPU operator
    EXEC_NPU_CMD(aclnnAddLayerNormGrad,
                 dy, x1, x2, rstd, mean, gamma, dsum_npu,
                 dxOut, dgammaOut, dbetaOut);

    std::vector<at::Tensor> result = {dxOut, dgammaOut, dbetaOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `add_layer_norm_grad`
    m.def("add_layer_norm_grad", &add_layer_norm_grad, "");
}