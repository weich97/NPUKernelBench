#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_fun(at::Tensor &dy, at::Tensor &x, at::Tensor &y)
{
    at::Tensor result = torch::empty_like(x);

    EXEC_NPU_CMD(aclnnCustomOp, dy, x, y, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("gelu_grad", &custom_pybind_fun, "");
}