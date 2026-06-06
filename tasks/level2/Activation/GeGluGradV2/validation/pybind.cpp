// pybind.py
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
 * Implementation note.
 * Implementation note.
 *
 * Implementation note.
 */
at::Tensor custom_pybind_api(at::Tensor dy, at::Tensor x, at::Tensor gelu, int64_t dim, int64_t approximate, bool activateLeft)
{
    // Implementation note.
    // The output shape of GeGluGradV2 typically matches the input 'dy' or 'x'.
    // Assuming output shape is the same as 'x' for this example.
    at::Tensor out = torch::empty_like(x);

    // Implementation note.
    // YYY=aclnnGeGluGradV2GetWorkspaceSize(const aclTensor *dy, const aclTensor *x, const aclTensor *gelu, int64_t dim, int64_t approximate, bool activateLeft, const aclTensor *out);
    EXEC_NPU_CMD(aclnnGeGluGradV2, dy, x, gelu, dim, approximate, activateLeft, out);

    return out;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("ge_glu_grad_v2", &custom_pybind_api, "");
}