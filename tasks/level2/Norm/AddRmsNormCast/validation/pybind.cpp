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
std::vector<at::Tensor> custom_pybind_api(at::Tensor x1, at::Tensor x2, at::Tensor gamma, double epsilon)
{   
    auto options = torch::TensorOptions().dtype(torch::kFloat).device(x1.device());
    at::Tensor y1 = torch::empty(x1.sizes(), options); // only float

    at::Tensor y2 = torch::empty_like(x1); // float16 or bf16
    at::Tensor x = torch::empty_like(x1);

    torch::IntArrayRef x_shapes = x.sizes();
    std::vector<int64_t> reduced_shape(x_shapes.begin(), x_shapes.end());
    reduced_shape[reduced_shape.size() - 1] = 1;
    
    at::Tensor rstdOut = torch::empty(reduced_shape, options); // only float

    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, epsilon, y1, y2, rstdOut, x);

    std::vector<at::Tensor> result = {y1, y2, rstdOut, x};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}