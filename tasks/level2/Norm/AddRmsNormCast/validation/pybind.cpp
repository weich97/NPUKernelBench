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
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(x1.device());
    at::Tensor y1 = torch::empty(x1.sizes(), options); // only float

    at::Tensor y2 = torch::empty_like(x1); // float16 or bf16
    at::Tensor x = torch::empty_like(x1);

    torch::IntArrayRef x_shapes = x.sizes();
    std::vector<int64_t> reduced_shape(x_shapes.begin(), x_shapes.end());
    reduced_shape[reduced_shape.size() - 1] = 1;
    
    at::Tensor rstdOut = torch::empty(reduced_shape, options); // only float

    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, epsilon, y1, y2, rstdOut, x);

    std::vector<at::Tensor> result = {y1, y2, rstdOut, x};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}