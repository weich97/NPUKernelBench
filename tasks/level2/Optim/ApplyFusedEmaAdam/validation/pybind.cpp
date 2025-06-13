#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
std::vector<at::Tensor> custom_pybind_fun(at::Tensor &grad, at::Tensor &varRef, at::Tensor &mRef,
                              at::Tensor &vRef, at::Tensor &sRef, at::Tensor &step,
                              double lr, double emaDecay, double beta1, double beta2,
                              double eps, int64_t mode, bool biasCorrection, double weightDecay)
{
    // 返回变量引用作为结果，实际上算子会原地修改这些引用
    EXEC_NPU_CMD(aclnnCustomOp, grad, varRef, mRef, vRef, sRef, step,
                 lr, emaDecay, beta1, beta2, eps, mode, biasCorrection, weightDecay);
    std::vector<at::Tensor> results = {varRef, mRef, vRef, sRef};
    return results;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_fun, "");
}