/**
 * @file extension_non_finite_check.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

// ... (comments as before)

at::TensorList to_tensor_list(const std::vector<at::Tensor>& vec) {
    return at::TensorList(vec);
}

at::Tensor non_finite_check_op(std::vector<at::Tensor> &x)
{
    // 输出张量列表，每个 input tensor 映射一个标量输出
    at::Tensor y = torch::empty({1}, x[0].options().dtype(torch::kFloat32));
    y.zero_();
    
    auto x_list = to_tensor_list(x);  // 先保存再传
    EXEC_NPU_CMD(aclnnNonFiniteCheckOp, x_list, y);

    return y;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.def("non_finite_check_op", &non_finite_check_op, "");
}