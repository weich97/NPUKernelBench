/**
 * @file extension_strided_slice_assign_v2.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp" // Implementation note.

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
at::Tensor custom_pybind_api(at::Tensor varRef, at::Tensor inputValue, at::Tensor begin, at::Tensor end, at::Tensor strides, at::Tensor axesOptional)
{
    // Implementation note.
    at::Tensor begin_cpu = begin.cpu();
    TORCH_CHECK(begin_cpu.dtype() == torch::kInt64, "begin tensor must be of type int64.");
    TORCH_CHECK(begin_cpu.dim() == 1, "begin tensor must be 1-dimensional.");
    std::vector<int64_t> begin_vec(begin_cpu.data_ptr<int64_t>(), begin_cpu.data_ptr<int64_t>() + begin_cpu.numel());

    // Implementation note.
    aclIntArray* begin_acl = ConvertType(at::IntArrayRef(begin_vec));

    // Implementation note.
    at::Tensor end_cpu = end.cpu();
    TORCH_CHECK(end_cpu.dtype() == torch::kInt64, "end tensor must be of type int64.");
    TORCH_CHECK(end_cpu.dim() == 1, "end tensor must be 1-dimensional.");
    std::vector<int64_t> end_vec(end_cpu.data_ptr<int64_t>(), end_cpu.data_ptr<int64_t>() + end_cpu.numel());
    aclIntArray* end_acl = ConvertType(at::IntArrayRef(end_vec));

    at::Tensor strides_cpu = strides.cpu();
    TORCH_CHECK(strides_cpu.dtype() == torch::kInt64, "strides tensor must be of type int64.");
    TORCH_CHECK(strides_cpu.dim() == 1, "strides tensor must be 1-dimensional.");
    std::vector<int64_t> strides_vec(strides_cpu.data_ptr<int64_t>(), strides_cpu.data_ptr<int64_t>() + strides_cpu.numel());
    aclIntArray* strides_acl = ConvertType(at::IntArrayRef(strides_vec));

    at::Tensor axesOptional_cpu = axesOptional.cpu();
    TORCH_CHECK(axesOptional_cpu.dtype() == torch::kInt64, "axesOptional tensor must be of type int64.");
    TORCH_CHECK(axesOptional_cpu.dim() == 1, "axesOptional tensor must be 1-dimensional.");
    std::vector<int64_t> axesOptional_vec(axesOptional_cpu.data_ptr<int64_t>(), axesOptional_cpu.data_ptr<int64_t>() + axesOptional_cpu.numel());
    aclIntArray* axes_optional_acl = ConvertType(at::IntArrayRef(axesOptional_vec));

    EXEC_NPU_CMD(aclnnStridedSliceAssignV2, varRef, inputValue, begin_acl, end_acl, strides_acl, axes_optional_acl);

    // Implementation note.
    return varRef;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("strided_slice_assign_v2", &custom_pybind_api, "");
}