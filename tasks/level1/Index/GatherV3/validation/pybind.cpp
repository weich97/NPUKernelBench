#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &self_tensor, at::Tensor &axis, at::Tensor &indices)
{

    // Compute the expected output shape
    auto self_sizes = self_tensor.sizes().vec();
    int64_t dim = axis.item<int64_t>();
    self_sizes[dim] = indices.size(0); // Replace the selected dim's size with indices.size(0)

    // Allocate output tensor with the correct shape
    at::Tensor result = torch::empty(self_sizes, self_tensor.options());

    EXEC_NPU_CMD(aclnnGatherV3, self_tensor, indices, axis, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}