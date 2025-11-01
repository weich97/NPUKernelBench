/**
 * @file extension_layer_norm_v4.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp" // Ensure this includes ConvertType and EXEC_NPU_CMD

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
std::vector<at::Tensor> layer_norm_v4(at::Tensor input, at::IntArrayRef normalized_shape, c10::optional<at::Tensor> weightOptional, c10::optional<at::Tensor> biasOptional, double eps)
{
    // Outputs: out, meanOutOptional, rstdOutOptional
    at::Tensor out = torch::empty_like(input);

    // meanOutOptional and rstdOutOptional's shape will have 1s in the normalized dimensions
    std::vector<int64_t> mean_rstd_shape(input.sizes().begin(), input.sizes().end());
    for(int i = 0; i < normalized_shape.size(); i++) {
        mean_rstd_shape[mean_rstd_shape.size() - 1 - i] = 1;
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(input.device()); // mean/rstd are usually float

    // Create optional tensors for mean and rstd. If not used, they will be empty.
    at::Tensor meanOut = torch::empty(mean_rstd_shape, options);
    at::Tensor rstdOut = torch::empty(mean_rstd_shape, options);

    // Convert normalized_shape to aclIntArray*
    aclIntArray* normalizedShape_acl = ConvertType(normalized_shape);

    // Prepare optional weight and bias tensors for NPU_CMD
    at::Tensor weight_npu = weightOptional.has_value() ? weightOptional.value() : at::Tensor();
    at::Tensor bias_npu = biasOptional.has_value() ? biasOptional.value() : at::Tensor();

    EXEC_NPU_CMD(aclnnLayerNorm, input, normalizedShape_acl, weight_npu, bias_npu, eps, out, meanOut, rstdOut);

    std::vector<at::Tensor> result = {out, meanOut, rstdOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `layer_norm_v4`
    m.def("layer_norm_v4", &layer_norm_v4, "");
}