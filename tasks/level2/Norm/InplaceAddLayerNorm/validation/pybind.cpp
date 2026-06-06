/**
 * @file extension_add.cpp
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
std::vector<at::Tensor> custom_pybind_api(at::Tensor x1, at::Tensor x2, c10::optional<at::Tensor> bias, at::Tensor gamma, at::Tensor beta, float epsilon, bool additionalOut)
{
    // Implementation note.
    torch::IntArrayRef shapes = x1.sizes();
    std::vector<int64_t> meanOutShape(shapes.begin(), shapes.end());
    for(int i = 0; i < gamma.sizes().size(); i++) {
        meanOutShape[meanOutShape.size() - i - 1] = 1;
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(x1.device());
    auto meanOut = torch::empty(meanOutShape, options);
    auto rstdOut = torch::empty_like(meanOut);

    if (bias.has_value()){
        EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, beta, bias, epsilon, additionalOut, meanOut, rstdOut);
    } else {
        at::Tensor empty_tensor = at::Tensor();
        EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, beta, empty_tensor, epsilon, additionalOut, meanOut, rstdOut);
    }
    std::vector<at::Tensor> result;
    if (additionalOut) {
        result = {x1, meanOut, rstdOut, x2};
    } else {
        result = {x1};
    }
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}