/**
 * @file extension_icamax.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * 使用说明：
 * 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnForeachExp
 * 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 foreach_exp
 *
 * 替换示例：
 * - aclnnCustomOp -> aclnnXXX (其中XXX为算子名，如替换为aclnnForeachExp)
 * - custom_pybind_api -> YYY (其中YYY为算子名的下划线形式，如foreach_exp)
 *
 * 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
 */
std::vector<at::Tensor> icamax(at::Tensor x, int64_t n, int64_t incx)
{
    // Output tensor 'out' will be a scalar (1-element) int32 tensor.
    at::Tensor out = torch::empty({}, x.options().dtype(torch::kInt32));

    // EXEC_NPU_CMD takes the actual NPU operator name, `aclnnIcamax`.
    EXEC_NPU_CMD(aclnnIcamax, x, n, incx, out);

    std::vector<at::Tensor> result = {out};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `icamax`
    m.def("icamax", &icamax, "");
}