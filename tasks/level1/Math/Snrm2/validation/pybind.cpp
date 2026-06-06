/**
 * @file extension_snrm2.cpp
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
at::Tensor snrm2(at::Tensor x, int64_t n, int64_t incx)
{
    // Output tensor 'out' will be a scalar (1-element) tensor with same dtype as x.
    at::Tensor out = torch::empty({}, x.options()); // Matches input dtype

    EXEC_NPU_CMD(aclnnSnrm2, x, n, incx, out);

    return out;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `snrm2`
    m.def("snrm2", &snrm2, "");
}