/**
 * @file extension_is_finite.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * Usage note: replace placeholder operator names while preserving function signatures.
 * Usage note: replace placeholder operator names while preserving function signatures.
 * Usage note: replace placeholder operator names while preserving function signatures.
 *
 * Usage note: replace placeholder operator names while preserving function signatures.
 * Usage note: replace placeholder operator names while preserving function signatures.
 * Usage note: replace placeholder operator names while preserving function signatures.
 *
 * Usage note: replace placeholder operator names while preserving function signatures.
 */
at::Tensor is_finite(at::Tensor x)
{
    // 1. Create an output tensor 'y_bool_output' with boolean data type.
    //    This is because aclnnIsFinite is expected to produce a boolean result.
    at::Tensor y_bool_output = torch::empty_like(x, x.options().dtype(torch::kBool));

    // 2. Execute the ACLNN operator. It will write boolean values (False/True) to y_bool_output.
    EXEC_NPU_CMD(aclnnIsFinite, x, y_bool_output);

    // 3. Convert the boolean output tensor to the desired numeric type (0.0 or 1.0).
    //    We convert it to the same dtype as the input 'x' for consistency.
    at::Tensor y_numeric_output = y_bool_output.to(x.dtype());

    return y_numeric_output;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `is_finite`
    m.def("is_finite", &is_finite, "");
}
