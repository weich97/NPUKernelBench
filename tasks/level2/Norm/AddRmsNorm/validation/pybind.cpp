/**
 * @file extension_add_rms_norm.cpp
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
std::vector<at::Tensor> add_rms_norm(at::Tensor x1, at::Tensor x2, at::Tensor gamma, double epsilonOptional)
{
    // Output tensors: yOut (RMSNorm result), rstdOut (reciprocal standard deviation), xOut (summed input)
    at::Tensor yOut = torch::empty_like(x1); // yOut has same shape as x1, x2
    
    // rstdOut's shape is typically the input shape with normalized dimensions set to 1.
    // Assuming normalization on the last dimension, rstdOut will have 1 in the last dimension.
    // For a 2D tensor (B, L) with gamma on L, rstd is (B, 1).
    // Let's deduce rstdOutShape based on gamma.dim() as normalized_shape.
    torch::IntArrayRef x1_shapes = x1.sizes();
    std::vector<int64_t> rstdOutShape(x1_shapes.begin(), x1_shapes.end());
    for(int i = 0; i < gamma.sizes().size(); i++) {
        rstdOutShape[rstdOutShape.size() - 1 - i] = 1; // Set normalized dims to 1
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(x1.device()); // rstd is usually float
    at::Tensor rstdOut = torch::empty(rstdOutShape, options);
    
    at::Tensor xOut = torch::empty_like(x1); // xOut is the sum x1 + x2

    EXEC_NPU_CMD(aclnnAddRmsNorm, x1, x2, gamma, epsilonOptional, yOut, rstdOut, xOut);

    std::vector<at::Tensor> result = {yOut, rstdOut, xOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `add_rms_norm`
    m.def("add_rms_norm", &add_rms_norm, "");
}