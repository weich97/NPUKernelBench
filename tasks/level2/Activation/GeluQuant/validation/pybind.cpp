#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> gelu_quant_wrapper(
    const at::Tensor& x,
    const at::Tensor& scale,
    const at::Tensor& offset,
    const std::string& approximate,
    const std::string& quant_mode)
{
    // 构建输出 tensor 和 outScale
    at::Tensor y = at::zeros_like(x, x.options().dtype(torch::kInt8));
    auto out_scale_shape = x.sizes().vec();
    if (!out_scale_shape.empty()) out_scale_shape.back() = 1;
    at::Tensor out_scale = at::zeros(out_scale_shape, x.options().dtype(torch::kFloat32));

    const char* approximate_ = approximate.c_str();
    const char* quant_mode_ = quant_mode.c_str();

    EXEC_NPU_CMD(aclnnGeluQuant, x, scale, offset, approximate_, quant_mode_, y, out_scale);

    return {y, out_scale};
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.def("gelu_quant", &gelu_quant_wrapper, "GELU Quant wrapper");
}
