#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * Implementation note.
 * Implementation note.
 * Implementation note.
 *
 * Implementation note.
 * - aclnnCustomOp -> aclnnGeGluV2
 * - custom_pybind_api -> ge_glu_v2
 *
 * Implementation note.
 */
std::vector<at::Tensor> custom_pybind_api(at::Tensor x)
{
    // Implementation note.
    auto x_sizes = x.sizes().vec(); // Implementation note.

    // Implementation note.
    TORCH_CHECK(x_sizes.back() % 2 == 0, "Last dimension of input x must be divisible by 2 for GeGluV2.");
    int64_t dim = -1;
    int64_t approximate = 0;
    bool activate_left = true;

    // Implementation note.
    x_sizes.back() /= 2;
    at::Tensor result = torch::empty(x_sizes, x.options());
    at::Tensor outGelu = torch::empty_like(result);
    size_t workspace_size = 0;

    // Implementation note.
    EXEC_NPU_CMD(aclnnGeGluV2, x, dim, approximate, activate_left, result, outGelu);

    // Return a std::vector containing the two tensors
    std::vector<at::Tensor> result_tensors;
    result_tensors.push_back(result);
    result_tensors.push_back(outGelu);    
    return result_tensors;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("ge_glu_v2", &custom_pybind_api, "");
}