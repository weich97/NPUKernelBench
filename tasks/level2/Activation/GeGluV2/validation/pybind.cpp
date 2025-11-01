#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * 使用说明：
 * 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnGeGluV2
 * 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 ge_glu_v2
 *
 * 替换示例：
 * - aclnnCustomOp -> aclnnGeGluV2
 * - custom_pybind_api -> ge_glu_v2
 *
 * 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
 */
std::vector<at::Tensor> custom_pybind_api(at::Tensor x)
{
    // 获取输入张量的形状
    auto x_sizes = x.sizes().vec();  // 转为 std::vector<int64_t>

    // 检查输入张量 x 的最后一维是否可以被 2 整除
    TORCH_CHECK(x_sizes.back() % 2 == 0, "Last dimension of input x must be divisible by 2 for GeGluV2.");
    int64_t dim = -1;
    int64_t approximate = 0;
    bool activate_left = true;

    // 创建输出张量 result，其形状为输入 x 的最后一维减半
    x_sizes.back() /= 2;
    at::Tensor result = torch::empty(x_sizes, x.options());
    at::Tensor outGelu = torch::empty_like(result);
    size_t workspace_size = 0;

    // 执行 NPU 自定义算子
    EXEC_NPU_CMD(aclnnGeGluV2, x, dim, approximate, activate_left, result, outGelu);

    // Return a std::vector containing the two tensors
    std::vector<at::Tensor> result_tensors;
    result_tensors.push_back(result);
    result_tensors.push_back(outGelu);    
    return result_tensors;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("ge_glu_v2", &custom_pybind_api, "");
}