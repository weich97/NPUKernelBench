#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &mul0_input0, at::Tensor &mul0_input1, at::Tensor &mul1_input0, at::Tensor &add_y, at::Tensor &gamma, at::Tensor &beta)
{
    at::Tensor result = torch::empty_like(mul0_input0);

    EXEC_NPU_CMD(aclnnCustomOp, mul0_input0, mul0_input1, mul1_input0, add_y, gamma, beta, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}