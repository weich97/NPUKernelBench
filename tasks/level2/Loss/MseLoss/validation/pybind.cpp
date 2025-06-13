#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_fun(at::Tensor &predict, at::Tensor &label, std::string reduction)
{
    at::Tensor result;
    int64_t reductionCode = 0;
    if (reduction == "none") {
        result = torch::empty_like(predict);
        reductionCode = 3;
    } else {
        result = at::empty({1}, predict.options());
        if (reduction == "mean") {
            reductionCode = 1;
        } else if (reduction == "sum") {
            reductionCode = 2;
        }
    }
    EXEC_NPU_CMD(aclnnCustomOp, predict, label, reductionCode, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_fun, "");
}