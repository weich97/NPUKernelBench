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
std::vector<at::Tensor> custom_pybind_api(std::vector<at::Tensor> &x, at::Tensor &scalar_tensor)
{
    // 从tensor中提取标量值
    std::vector<at::Scalar> scalars;
    scalars.reserve(scalar_tensor.size(0));

    // 根据标量张量的数据类型进行处理
    if (scalar_tensor.scalar_type() == at::ScalarType::Float) {
        auto accessor = scalar_tensor.accessor<float, 1>();
        for (int i = 0; i < scalar_tensor.size(0); i++) {
            scalars.push_back(at::Scalar(accessor[i]));
        }
    }
    else if (scalar_tensor.scalar_type() == at::ScalarType::Double) {
        auto accessor = scalar_tensor.accessor<double, 1>();
        for (int i = 0; i < scalar_tensor.size(0); i++) {
            scalars.push_back(at::Scalar(accessor[i]));
        }
    }
    else if (scalar_tensor.scalar_type() == at::ScalarType::Int) {
        auto accessor = scalar_tensor.accessor<int, 1>();
        for (int i = 0; i < scalar_tensor.size(0); i++) {
            scalars.push_back(at::Scalar(accessor[i]));
        }
    }
    else if (scalar_tensor.scalar_type() == at::ScalarType::Long) {
        auto accessor = scalar_tensor.accessor<int64_t, 1>();
        for (int i = 0; i < scalar_tensor.size(0); i++) {
            scalars.push_back(at::Scalar(accessor[i]));
        }
    }
    else {
        throw std::runtime_error("Unsupported scalar type for ForeachAddScalarList");
    }

    // 其余代码不变
    std::vector<at::Tensor> result;
    result.reserve(x.size());

    for (const auto &tensor : x) {
        result.push_back(torch::empty_like(tensor));
    }

    at::TensorList result_list = at::TensorList(result);
    at::TensorList input = at::TensorList(x);
    at::ArrayRef<at::Scalar> scalar_list(scalars);

    EXEC_NPU_CMD(aclnnCustomOp, input, scalar_list, result_list);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}