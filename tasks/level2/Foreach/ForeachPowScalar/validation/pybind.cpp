#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * 使用说明：
 * 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnForeachPowScalar
 * 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 foreach_pow_scalar
 *
 * 替换示例：
 * - aclnnCustomOp -> aclnnForeachPowScalar
 * - custom_pybind_api -> foreach_pow_scalar
 *
 * 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
 */
#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"   // 包含 EXEC_NPU_CMD 宏

std::vector<at::Tensor> custom_pybind_api(std::vector<at::Tensor> &x, at::Tensor &scalar)
{
    // alloc output memory
    std::vector<at::Tensor> result;
    result.reserve(x.size());

    for (const auto &tensor : x) {
        result.push_back(torch::empty_like(tensor));
    }

    // Convert std::vector<at::Tensor> to at::TensorList for the EXEC_NPU_CMD
    at::TensorList result_list = at::TensorList(result);
    at::TensorList x_list = at::TensorList(x);

    // 直接传引用
    EXEC_NPU_CMD(aclnnForeachPowScalar, x_list, scalar, result_list);

    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("foreach_pow_scalar", &custom_pybind_api, "");
}