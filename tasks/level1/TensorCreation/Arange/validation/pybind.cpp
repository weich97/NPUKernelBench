#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor custom_pybind_api(at::Tensor &start, at::Tensor &end, at::Tensor &step)
{

    at::ScalarType scalar_type = start.scalar_type();
    aclScalar* scalarStart = nullptr;
    aclScalar* scalarEnd = nullptr;
    aclScalar* scalarStep = nullptr;
    // 提前声明变量
    int value_start = 0.0;
    int value_end = 0.0;
    int value_step = 0.0;

    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);

    switch (scalar_type) {
    case at::ScalarType::Half: {
        // Extract the scalar value as float and cast to aclFloat16
        value_start = static_cast<aclFloat16>(start.item<float>());
        value_end = static_cast<aclFloat16>(end.item<float>());
        value_step = static_cast<aclFloat16>(step.item<float>());
        scalarStart = aclCreateScalar(&value_start, aclDataType::ACL_FLOAT16);
        scalarEnd = aclCreateScalar(&value_end, aclDataType::ACL_FLOAT16);
        scalarStep = aclCreateScalar(&value_step, aclDataType::ACL_FLOAT16);
        break;
    }
    case at::ScalarType::Float: {
        // Extract the scalar value as float
        value_start = start.item<float>();
        value_end = end.item<float>();
        value_step = step.item<float>();
        scalarStart = aclCreateScalar(&value_start, aclDataType::ACL_FLOAT);
        scalarEnd = aclCreateScalar(&value_end, aclDataType::ACL_FLOAT);
        scalarStep = aclCreateScalar(&value_step, aclDataType::ACL_FLOAT);
        break;
    }
    case at::ScalarType::Int: {
        // Extract the scalar value as int

        value_start = start.item<int>();
        value_end = end.item<int>();
        value_step = step.item<int>();
        scalarStart = aclCreateScalar(&value_start, aclDataType::ACL_INT32);
        scalarEnd = aclCreateScalar(&value_end, aclDataType::ACL_INT32);
        scalarStep = aclCreateScalar(&value_step, aclDataType::ACL_INT32);

        std::cout << "value_start = " << value_start << std::endl;

        break;
    }
    default:
        // Raise an error for unsupported scalar types
        TORCH_CHECK(false, "Unsupported scalar type: ", toString(scalar_type));
    }

    double size_arange = ceil(static_cast<double>(value_end - value_start) / value_step);
    int64_t size_value = static_cast<int64_t>(size_arange);
    
    at::Tensor result = torch::empty(size_value, start.options());

    EXEC_NPU_CMD(aclnnCustomOp, scalarStart, scalarEnd, scalarStep, result);

    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}







