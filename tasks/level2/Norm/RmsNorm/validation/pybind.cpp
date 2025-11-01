/**
 * @file extension_rms_norm.cpp
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
std::vector<at::Tensor> rms_norm(at::Tensor x, at::Tensor gamma, float epsilon)
{
    // 其余代码不变
    torch::IntArrayRef shapes = x.sizes();
    std::vector<int64_t> rstdOutShape(shapes.begin(), shapes.end());
    for(int i = 0; i < gamma.sizes().size(); i++) {
        rstdOutShape[rstdOutShape.size() - i - 1] = 1;
    }
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(x.device());
    auto rstdOut = torch::empty(rstdOutShape, options);
    at::Tensor yOut = torch::empty_like(x);

    EXEC_NPU_CMD(aclnnRmsNorm, x, gamma, epsilon, yOut, rstdOut);

    std::vector<at::Tensor> result = {yOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("rms_norm", &rms_norm, "");
}