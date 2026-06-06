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
at::Tensor ccopy(at::Tensor x, int64_t n, int64_t incx, int64_t incy)
{
    at::Tensor out = torch::empty({n * incy}, x.options());
    EXEC_NPU_CMD(aclnnCcopy, x, n, incx, incy, out);

    return out;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("ccopy", &ccopy, "");
}