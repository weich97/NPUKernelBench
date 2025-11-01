/**
 * @file extension_strided_slice_assign_v2.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp" // 确保包含了此头文件，其中定义了 ConvertType 和 EXEC_NPU_CMD

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
at::Tensor custom_pybind_api(at::Tensor varRef, at::Tensor inputValue, at::Tensor begin, at::Tensor end, at::Tensor strides, at::Tensor axesOptional)
{
    // 将 begin_cpu (at::Tensor) 转换为 std::vector<int64_t>
    at::Tensor begin_cpu = begin.cpu();
    TORCH_CHECK(begin_cpu.dtype() == torch::kInt64, "begin tensor must be of type int64.");
    TORCH_CHECK(begin_cpu.dim() == 1, "begin tensor must be 1-dimensional.");
    std::vector<int64_t> begin_vec(begin_cpu.data_ptr<int64_t>(), begin_cpu.data_ptr<int64_t>() + begin_cpu.numel());

    // 显式构造 at::IntArrayRef 并转换为 aclIntArray*
    aclIntArray* begin_acl = ConvertType(at::IntArrayRef(begin_vec));

    // 对其他参数也进行同样处理
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

    // 返回 varRef，因为它是一个原地修改的操作
    return varRef;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("strided_slice_assign_v2", &custom_pybind_api, "");
}