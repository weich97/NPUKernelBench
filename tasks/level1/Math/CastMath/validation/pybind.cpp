#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include <map>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

 std::map<int64_t, torch::Dtype> type_map = {
    {0, torch::kFloat32},
    {1, torch::kFloat16},
    {2, torch::kInt8},
    {3, torch::kInt32},
    {4, torch::kUInt8},
    {6, torch::kInt16},
    {9, torch::kInt64},
    {12, torch::kBool},
    {27, torch::kBFloat16}
};

at::Tensor custom_pybind_api(at::Tensor &x, int64_t dst_type)
{
    auto result_options = torch::TensorOptions()
                              .dtype(type_map.at(dst_type))
                              .device(x.device());

    // Allocate the output tensor 'result' with the same dimensions as 'x' but with the new data type.
    at::Tensor result = torch::empty(x.sizes(), result_options);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnCustomOp, x, dst_type, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}