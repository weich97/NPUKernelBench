/**
 * @file extension_add.cpp
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
std::vector<at::Tensor> custom_pybind_api(
    at::Tensor x1, 
    at::Tensor x2, 
    at::Tensor gamma, 
    at::Tensor scales1, 
    c10::optional<at::Tensor> scales2, 
    c10::optional<at::Tensor> zero_points1,
    c10::optional<at::Tensor> zero_points2,
    int64_t axis = -1,
    float epsilon = 1e-5f, 
    bool div_mode = true )
{
    // 其余代码不变
    torch::IntArrayRef shapes = x1.sizes();
    std::vector<int64_t> meanOutShape(shapes.begin(), shapes.end());
    for(int i = 0; i < gamma.sizes().size(); i++) {
        meanOutShape[meanOutShape.size() - i - 1] = 1;
    }
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(x1.device()); // rstd is usually float
    at::Tensor yOut1 = torch::empty_like(x1, options);
    at::Tensor yOut2 = torch::empty_like(x1, options);

    at::Tensor xOut = torch::empty_like(x1);
   
    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, axis, epsilon, div_mode, yOut1, yOut2, xOut);
    
    std::vector<at::Tensor> result;
    result = {yOut1};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}