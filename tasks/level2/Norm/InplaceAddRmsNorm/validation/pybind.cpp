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
std::vector<at::Tensor> custom_pybind_api(at::Tensor x1, at::Tensor x2, at::Tensor gamma, double epsilon)
{   
    at::Tensor yOut = torch::empty_like(x1); // yOut has same shape as x1, x2

    torch::IntArrayRef x1_shapes = x1.sizes();
    std::vector<int64_t> rstdOutShape(x1_shapes.begin(), x1_shapes.end());
    for(int i = 0; i < gamma.sizes().size(); i++) {
        rstdOutShape[rstdOutShape.size() - 1 - i] = 1; // Set normalized dims to 1
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(x1.device()); // rstd is usually float
    at::Tensor rstdOut = torch::empty(rstdOutShape, options);

    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, epsilon, rstdOut);

    std::vector<at::Tensor> result = {x1, rstdOut, x2};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}