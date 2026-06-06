/**
 * @file extension_is_inf.cpp
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
std::vector<at::Tensor> is_inf(at::Tensor x)
{
    // 1. Create an output tensor 'y_bool_output' with boolean data type.
    //    This is because aclnnIsInf is expected to produce a boolean result.
    at::Tensor y_bool_output = torch::empty_like(x, x.options().dtype(torch::kBool));

    // 2. Execute the ACLNN operator. It will write boolean values (False/True) to y_bool_output.
    EXEC_NPU_CMD(aclnnIsInf, x, y_bool_output);

    // 3. Convert the boolean output tensor to the desired numeric type (e.g., float16, float32, bfloat16)
    //    where False becomes 0.0 and True becomes 1.0.
    //    We convert it to the same dtype as the input 'x' for consistency.
    at::Tensor y_numeric_output = y_bool_output.to(x.dtype());

    std::vector<at::Tensor> result = {y_numeric_output};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `is_inf`
    m.def("is_inf", &is_inf, "");
}
