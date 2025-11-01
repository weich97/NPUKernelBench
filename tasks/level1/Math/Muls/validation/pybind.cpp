#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor custom_pybind_api(at::Tensor &x, at::Tensor &value)
{
    // alloc output memory
    at::Tensor result = torch::empty_like(x);

    c10::Scalar base_scalar(value.item<float>());

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnCustomOp, x, base_scalar, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}