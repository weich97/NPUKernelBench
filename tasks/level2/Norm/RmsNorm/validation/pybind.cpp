/**
 * @file extension_rms_norm.cpp
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
std::vector<at::Tensor> rms_norm(at::Tensor x, at::Tensor gamma, float epsilon)
{
    // Implementation note.
    torch::IntArrayRef shapes = x.sizes();
    std::vector<int64_t> rstdOutShape(shapes.begin(), shapes.end());
    for(int i = 0; i < gamma.sizes().size(); i++) {
        rstdOutShape[rstdOutShape.size() - i - 1] = 1;
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(x.device());
    auto rstdOut = torch::empty(rstdOutShape, options);
    at::Tensor yOut = torch::empty_like(x);

    EXEC_NPU_CMD(aclnnRmsNorm, x, gamma, epsilon, yOut, rstdOut);

    std::vector<at::Tensor> result = {yOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("rms_norm", &rms_norm, "");
}