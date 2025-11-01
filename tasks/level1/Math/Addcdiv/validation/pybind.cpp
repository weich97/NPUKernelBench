#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &input_data, at::Tensor &input_x1, at::Tensor &input_x2, at::Tensor &value_tensor)
{
    at::Tensor result = torch::empty_like(input_data);
    auto scalarValue = ConvertTensorToAclScaler(value_tensor);

    EXEC_NPU_CMD(aclnnCustomOp, input_data, input_x1, input_x2, scalarValue, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}