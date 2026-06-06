/**
 * @file extension_mse_loss_grad.cpp
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
std::vector<at::Tensor> mse_loss_grad(at::Tensor gradOutput, at::Tensor self_input, at::Tensor target_input, std::string reduction)
{
    // Output tensor: out
    at::Tensor out = torch::empty_like(gradOutput);

    // Convert reduction string to const char*
    const char* reduction_cstr = reduction.c_str();

    int64_t reduction_int;
    if (reduction == "none") {
        reduction_int = 0;
    } else if (reduction == "mean") {
        reduction_int = 1;
    } else if (reduction == "sum") {
        reduction_int = 2;
    } else {
        TORCH_CHECK(false, "Unsupported reduction type: ", reduction);
    }

    EXEC_NPU_CMD(aclnnMseLossBackward, gradOutput, self_input, target_input, reduction_int, out);

    std::vector<at::Tensor> result = {out};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `mse_loss_grad`
    m.def("mse_loss_grad", &mse_loss_grad, "");
}