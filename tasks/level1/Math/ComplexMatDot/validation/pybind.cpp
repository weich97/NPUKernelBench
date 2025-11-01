#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor custom_pybind_api(at::Tensor &matx, at::Tensor &maty, int64_t m, int64_t n)
{
    // alloc output memory
    at::Tensor result = torch::empty_like(matx);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnComplexMatDot, matx, maty, m, n, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}