#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <ATen/Scalar.h> // For at::Scalar
#include <vector> // For std::vector

#include "pytorch_npu_helper.hpp" // Assuming this contains EXEC_NPU_CMD macro and relevant ACLNN headers

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
        std::vector<at::Tensor>& x1,
        std::vector<at::Tensor>& x2,
        at::Tensor value_tensor) {

    std::vector<at::Tensor> y_list;
    y_list.reserve(x1.size());
    for (const auto& t : x1) y_list.emplace_back(torch::zeros_like(t));

    // 直接构造 Scalar
    at::TensorList input1 = at::TensorList(x1);
    at::TensorList input2 = at::TensorList(x2);
    auto scalarValue = ConvertTensorToAclScaler(value_tensor);
    at::TensorList output = at::TensorList(y_list);
    EXEC_NPU_CMD(aclnnForeachLerpScalar, input1, input2, scalarValue, output);
    return y_list;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Define the Python binding for foreach_lerp_scalar
    m.def("foreach_lerp_scalar", &custom_pybind_api, "Performs foreach linear interpolation with a scalar weight.");
}