/**
 * @file extension_deep_norm_grad.cpp
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
std::vector<at::Tensor> deep_norm_grad(at::Tensor dy, at::Tensor x, at::Tensor gx, at::Tensor gamma, at::Tensor mean, at::Tensor rstd, double alpha)
{
    // Output tensors: dxOut, dgxOut, dbetaOut, dgammaOut
    at::Tensor dxOut = torch::empty_like(x);
    at::Tensor dgxOut = torch::empty_like(gx);
    at::Tensor dbetaOut = torch::empty_like(gamma, torch::dtype(torch::kFloat32)); // dbeta and dgamma have same shape as beta/gamma
    at::Tensor dgammaOut = torch::empty_like(gamma, torch::dtype(torch::kFloat32));

    EXEC_NPU_CMD(aclnnDeepNormGrad, dy, x, gx, gamma, mean, rstd, alpha, dxOut, dgxOut, dbetaOut, dgammaOut);

    std::vector<at::Tensor> result = {dxOut, dgxOut, dbetaOut, dgammaOut};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `deep_norm_grad`
    m.def("deep_norm_grad", &deep_norm_grad, "");
}