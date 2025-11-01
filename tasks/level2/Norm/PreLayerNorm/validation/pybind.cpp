#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &x, at::Tensor &y, at::Tensor &gamma, at::Tensor &beta,  double &epsilon)
{
    at::Tensor result = torch::empty_like(x);

    EXEC_NPU_CMD(aclnnCustomOp, x, y, gamma, beta, epsilon, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}