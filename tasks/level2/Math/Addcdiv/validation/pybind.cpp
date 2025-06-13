#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &input_data, at::Tensor &input_x1, at::Tensor &input_x2, at::Tensor &value_tensor)
{
    at::Tensor result = torch::empty_like(input_data);
    at::ScalarType scalar_type = value_tensor.scalar_type();
    aclScalar* scalarValue = nullptr;
    static const auto aclCreateScalar = GET_OP_API_FUNC(aclCreateScalar);
    // Handle scalar type cases
    switch (scalar_type) {
    case at::ScalarType::Half: {
        // Extract the scalar value as float and cast to aclFloat16
        aclFloat16 value_fp16 = static_cast<aclFloat16>(value_tensor.item<float>());
        scalarValue = aclCreateScalar(&value_fp16, aclDataType::ACL_FLOAT16);
        break;
    }
    case at::ScalarType::Float: {
        // Extract the scalar value as float
        float value_fp32 = value_tensor.item<float>();
        scalarValue = aclCreateScalar(&value_fp32, aclDataType::ACL_FLOAT);
        break;
    }
    case at::ScalarType::Int: {
        // Extract the scalar value as int
        int value_int32 = value_tensor.item<int>();
        scalarValue = aclCreateScalar(&value_int32, aclDataType::ACL_INT32);
        break;
    }
    default:
        // Raise an error for unsupported scalar types
        TORCH_CHECK(false, "Unsupported scalar type: ", toString(scalar_type));
    }

    EXEC_NPU_CMD(aclnnCustomOp, input_data, input_x1, input_x2, scalarValue, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}