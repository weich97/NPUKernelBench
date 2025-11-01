#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_fun(at::Tensor &predict, at::Tensor &label, std::string &reduction)
{
    at::Tensor result;
    const char* reduction_cstr = reduction.c_str();
    if (reduction == "none") {
        result = torch::empty_like(predict);
    } else {
        result = at::empty({1}, predict.options());
    }
    EXEC_NPU_CMD(aclnnCustomOp, predict, label, reduction_cstr, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_fun, "");
}