/**
 * @file extension_group_norm_swish.cpp
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
std::vector<at::Tensor> group_norm_swish(
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    int64_t numGroups,
    std::string dataFormatOptional, // char* dataFormatOptional
    double eps,
    bool activateSwish,
    double swishScale)
{
    // Output tensors: yOut, meanOut, rstdOut
    at::Tensor yOut = torch::empty_like(x);

    // meanOut and rstdOut usually have shape (N, num_groups, 1, 1, ...)
    // Or more specifically: (N, num_groups, 1, ..., 1) depending on input shape.
    // In GroupNorm, mean/rstd have shape (N, C/G, 1, ..., 1) where C/G is channels per group.
    // The `native_group_norm` also returns mean/rstd of shape (N, C/G), not (N, C/G, 1, 1, ...).
    // Let's assume they are `(N, C/G)` as in PyTorch's native_group_norm's second and third return values.
    // If x is (N, C, H, W) and groups G, then mean/rstd are (N, G).
    // The golden function has `input_rstd = np.array([0, 1, 2], dtype=np.float32).reshape(3, 1, 1)`
    // This implies `x_shape` could be `(3, C, HxW)` and `mean/rstd` are `(3, 1, 1)` or `(3, 1)`
    
    // For GroupNorm, the statistics are calculated per group.
    // The output shape for mean/rstd is (N, num_groups) if C % G == 0.
    // Or (N, num_channels / num_groups) if C is num_channels.
    
    // Let's deduce shape for meanOut and rstdOut:
    // x.sizes() is (N, C, H, W)
    // C is numChannels. Groups is numGroups.
    // Mean/rstd are computed for each group *across* spatial dims and channels *within* the group.
    // Their shape should be (N, numGroups).
    
    // PyTorch's aten.native_group_norm returns mean/rstd as (N, group)
    // e.g. if x is (N, C, H, W) and G groups, mean/rstd are (N, G).
    at::Tensor meanOut = torch::empty({x.size(0), numGroups}, x.options());
    at::Tensor rstdOut = torch::empty({x.size(0), numGroups}, x.options());


    // Convert reduction string to const char* if aclnn requires it, or handle directly.
    const char* data_format_cstr = dataFormatOptional.c_str();

    EXEC_NPU_CMD(aclnnGroupNormSwish,
                 x, gamma, beta, numGroups, data_format_cstr,
                 eps, activateSwish, swishScale,
                 yOut, meanOut, rstdOut);

    std::vector<at::Tensor> result = {yOut, meanOut, rstdOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `group_norm_swish`
    m.def("group_norm_swish", &group_norm_swish, "");
}