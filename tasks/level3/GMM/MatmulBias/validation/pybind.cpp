#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_fun(at::Tensor &a, at::Tensor &b, at::Tensor &bias)
{
    int64_t m = a.size(0);
    int64_t n = b.size(1);
    at::Tensor result = torch::empty({m, n}, a.options());

    EXEC_NPU_CMD(aclnnCustomOp, a, b, bias, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_fun, "");
}