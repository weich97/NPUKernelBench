#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * 使用说明：
 * 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnReverseSequence
 * 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 reverse_sequence
 *
 * 替换示例：
 * - aclnnCustomOp -> aclnnReverseSequence
 * - custom_pybind_api -> reverse_sequence
 *
 * 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
 */
at::Tensor custom_pybind_api(at::Tensor x, at::Tensor seqLengths, int64_t seqDim, int64_t batchDim)
{
    // 获取输入张量的形状
    auto x_sizes = x.sizes().vec();
    auto seqLengths_sizes = seqLengths.sizes().vec();

    // 检查 seqLengths 的形状
    TORCH_CHECK(seqLengths_sizes.size() == 1, "seqLengths must be a 1-D tensor.");
    TORCH_CHECK(seqLengths_sizes[0] == x_sizes[batchDim], "The size of seqLengths must be equal to the batch size.");
    TORCH_CHECK(seqDim >= 0 && seqDim < x_sizes.size(), "seqDim out of range");
    TORCH_CHECK(batchDim >= 0 && batchDim < x_sizes.size() && batchDim != seqDim, "batchDim out of range or equal to seqDim");

    // 创建输出张量
    at::Tensor result = torch::empty_like(x);

    // 执行 NPU 自定义算子
    EXEC_NPU_CMD(aclnnReverseSequence, x, seqLengths, seqDim, batchDim, result);

    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("reverse_sequence", &custom_pybind_api, "");
}