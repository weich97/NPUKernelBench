/**
 * @file extension_mse_loss_grad.cpp
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
std::vector<at::Tensor> mse_loss_grad(at::Tensor gradOutput, at::Tensor self_input, at::Tensor target_input, std::string reduction)
{
    // Output tensor: out
    at::Tensor out = torch::empty_like(gradOutput);

    // Convert reduction string to const char*
    const char* reduction_cstr = reduction.c_str();

    int64_t reduction_int;
    if (reduction == "none") {
        reduction_int = 0;
    } else if (reduction == "mean") {
        reduction_int = 1;
    } else if (reduction == "sum") {
        reduction_int = 2;
    } else {
        TORCH_CHECK(false, "Unsupported reduction type: ", reduction);
    }

    EXEC_NPU_CMD(aclnnMseLossBackward, gradOutput, self_input, target_input, reduction_int, out);

    std::vector<at::Tensor> result = {out};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `mse_loss_grad`
    m.def("mse_loss_grad", &mse_loss_grad, "");
}