/**
 * @file extension_icamax.cpp
 */
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
std::vector<at::Tensor> icamax(at::Tensor x, int64_t n, int64_t incx)
{
    // Output tensor 'out' will be a scalar (1-element) int32 tensor.
    at::Tensor out = torch::empty({}, x.options().dtype(torch::kInt32));

    // EXEC_NPU_CMD takes the actual NPU operator name, `aclnnIcamax`.
    EXEC_NPU_CMD(aclnnIcamax, x, n, incx, out);

    std::vector<at::Tensor> result = {out};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `icamax`
    m.def("icamax", &icamax, "");
}