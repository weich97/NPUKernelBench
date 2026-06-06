/**
 * @file extension_cross_entropy_loss_grad.cpp
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
std::vector<at::Tensor> cross_entropy_loss_grad(
    at::Tensor gradLoss,
    at::Tensor logProb,
    at::Tensor target,
    c10::optional<at::Tensor> weightOptional,
    c10::optional<at::Tensor> gradZlossOptional,
    c10::optional<at::Tensor> lseForZlossOptional,
    std::string reductionOptional,
    int64_t ignoreIndex,
    double labelSmoothing,
    double lseSquareScaleForZloss)
{
    // The output is typically grad_input, which has the same shape as the original input to CrossEntropyLoss
    at::Tensor gradInput = torch::empty_like(logProb);

    // Prepare optional tensors for NPU_CMD
    at::Tensor weight_npu = weightOptional.has_value() ? weightOptional.value() : at::Tensor();
    at::Tensor grad_zloss_npu = gradZlossOptional.has_value() ? gradZlossOptional.value() : at::Tensor();
    at::Tensor lse_for_zloss_npu = lseForZlossOptional.has_value() ? lseForZlossOptional.value() : at::Tensor();

    const char* reduction_cstr = reductionOptional.c_str();

    // Call the NPU operator
    EXEC_NPU_CMD(aclnnCrossEntropyLossGrad,
                 gradLoss, logProb, target, weight_npu, grad_zloss_npu, lse_for_zloss_npu,
                 reduction_cstr, ignoreIndex, labelSmoothing, lseSquareScaleForZloss,
                 gradInput);

    std::vector<at::Tensor> result = {gradInput};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `cross_entropy_loss_grad`
    m.def("cross_entropy_loss_grad", &cross_entropy_loss_grad, "");
}