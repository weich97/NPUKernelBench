// pybind.py
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
at::Tensor custom_pybind_api(at::Tensor dy, at::Tensor x, at::Tensor gelu, int64_t dim, int64_t approximate, bool activateLeft)
{
    // 创建 result 张量
    // The output shape of GeGluGradV2 typically matches the input 'dy' or 'x'.
    // Assuming output shape is the same as 'x' for this example.
    at::Tensor out = torch::empty_like(x);

    // 执行 NPU 自定义算子
    // YYY=aclnnGeGluGradV2GetWorkspaceSize(const aclTensor *dy, const aclTensor *x, const aclTensor *gelu, int64_t dim, int64_t approximate, bool activateLeft, const aclTensor *out);
    EXEC_NPU_CMD(aclnnGeGluGradV2, dy, x, gelu, dim, approximate, activateLeft, out);

    return out;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("ge_glu_grad_v2", &custom_pybind_api, "");
}