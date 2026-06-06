#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <ATen/Scalar.h> // For at::Scalar
#include <vector> // For std::vector

#include "pytorch_npu_helper.hpp" // Assuming this contains EXEC_NPU_CMD macro and relevant ACLNN headers

/**
 * register forward implementation for NPU device
 * Implementation note.
 * Implementation note.
 * Implementation note.
 *
 * Implementation note.
 * Implementation note.
 * Implementation note.
 *
 * Implementation note.
 */
std::vector<at::Tensor> custom_pybind_api(
        std::vector<at::Tensor>& x1,
        std::vector<at::Tensor>& x2,
        at::Tensor value_tensor) {

    std::vector<at::Tensor> y_list;
    y_list.reserve(x1.size());
    for (const auto& t : x1) y_list.emplace_back(torch::zeros_like(t));

    // Implementation note.
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