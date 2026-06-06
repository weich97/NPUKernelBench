/**
 * @file extension_deep_norm.cpp
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
std::vector<at::Tensor> deep_norm(at::Tensor x, at::Tensor gx, at::Tensor beta, at::Tensor gamma, double alphaOptional, double epsilonOptional)
{
    // Output tensors: meanOut, rstdOut, yOut
    // meanOut and rstdOut have shape of x with normalized dims (typically last) set to 1
    // yOut has same shape as x, gx
    at::Tensor yOut = torch::empty_like(x);

    torch::IntArrayRef x_shapes = x.sizes();
    std::vector<int64_t> reduced_shape(x_shapes.begin(), x_shapes.end());
    // Assuming normalization is on the last dimension, consistent with gen_golden_data_simple
    reduced_shape[reduced_shape.size() - 1] = 1;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(x.device()); // mean/rstd are usually float
    at::Tensor meanOut = torch::empty(reduced_shape, options);
    at::Tensor rstdOut = torch::empty(reduced_shape, options);
    
    EXEC_NPU_CMD(aclnnDeepNorm, x, gx, beta, gamma, alphaOptional, epsilonOptional, meanOut, rstdOut, yOut);

    std::vector<at::Tensor> result = {meanOut, rstdOut, yOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `deep_norm`
    m.def("deep_norm", &deep_norm, "");
}