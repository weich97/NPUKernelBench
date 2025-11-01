#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp" // Assuming this contains EXEC_NPU_CMD macro and necessary headers for ACLNN

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
std::vector<at::Tensor> custom_pybind_api(at::Tensor x,
                                          const c10::optional<at::Tensor>& weight_scale_optional,
                                          const c10::optional<at::Tensor>& activation_scale_optional,
                                          const c10::optional<at::Tensor>& bias_optional,
                                          const c10::optional<at::Tensor>& quant_scale_optional,
                                          const c10::optional<at::Tensor>& quant_offset_optional,
                                          const c10::optional<at::Tensor>& group_index_optional,
                                          bool activate_left, const std::string& quant_mode) {
    // Check for input x's last dimension
    TORCH_CHECK(x.dim() > 1 && x.sizes().back() % 2 == 0,
                "Input x must have more than one dimension and its last dimension must be divisible by 2 for DequantSwigluQuant.");

    // Calculate output shape for y
    std::vector<int64_t> y_out_sizes = x.sizes().vec();
    y_out_sizes.back() /= 2;

    // Create output tensor y (always int8 as per spec)
    at::Tensor y = torch::empty(y_out_sizes, x.options().dtype(torch::kInt8));

    // Handle scale output for dynamic quantization
    at::Tensor scale;
    int64_t scale_dim0_size = 1;
    for (size_t i = 0; i < x.dim() - 1; ++i) {
        scale_dim0_size *= x.sizes()[i];
    }
    scale = torch::empty({scale_dim0_size}, x.options().dtype(torch::kFloat));

    // Access optional tensors
    at::Tensor weight_scale_tensor = weight_scale_optional.value_or(at::Tensor());
    at::Tensor activation_scale_tensor = activation_scale_optional.value_or(at::Tensor());
    at::Tensor bias_tensor = bias_optional.value_or(at::Tensor());
    at::Tensor quant_scale_tensor = quant_scale_optional.value_or(at::Tensor());
    at::Tensor quant_offset_tensor = quant_offset_optional.value_or(at::Tensor());
    at::Tensor group_index_tensor = group_index_optional.value_or(at::Tensor());

    // Convert quant_mode string to C-style string for ACLNN
    const char* quant_mode_cstr = quant_mode.c_str();

    // Execute NPU custom operator
    EXEC_NPU_CMD(aclnnDequantSwigluQuant,
                 x,
                 weight_scale_tensor,
                 activation_scale_tensor,
                 bias_tensor,
                 quant_scale_tensor,
                 quant_offset_tensor,
                 group_index_tensor,
                 activate_left,
                 quant_mode_cstr, // Pass C-style string
                 y,                // Output Y
                 scale  // Output Scale (nullptr if not dynamic quant for int32)
                 );

    // Return as std::vector<at::Tensor>
    std::vector<at::Tensor> results{y, scale};
    return results;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Define the Python binding for dequant_swiglu_quant
    m.def("dequant_swiglu_quant", &custom_pybind_api,
          "Performs dequantization, SwiGLU, and requantization operation.");
}