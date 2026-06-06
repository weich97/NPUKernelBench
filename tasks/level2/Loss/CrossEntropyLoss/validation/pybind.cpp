// pybind.py
/**
 * @file extension_cross_entropy_loss.cpp
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
std::vector<at::Tensor> cross_entropy_loss(
    at::Tensor input,
    at::Tensor target,
    c10::optional<at::Tensor> weightOptional,
    std::string reductionOptional, // char* reductionOptional
    int64_t ignoreIndex,
    double labelSmoothing,
    double lseSquareScaleForZloss,
    bool returnZloss)
{
    // Output tensors: lossOut, logProbOut, zlossOut, lseForZlossOut
    
    // logProbOut has same shape as input
    at::Tensor logProbOut = torch::empty_like(input);

    // lossOut shape depends on reduction
    at::Tensor lossOut;
    if (reductionOptional == "none") {
        lossOut = torch::empty_like(target, input.options()); // loss is per-sample
    } else { // "mean" or "sum"
        lossOut = torch::empty({}, input.options()); // scalar output
    }

    // --- START CHANGE ---
    // Determine the dtype for zlossOut and lseForZlossOut based on input_dtype
    at::ScalarType zloss_output_dtype = input.scalar_type();
    // Although the general rule is to match input dtype, the support info is explicit:
    // SupportInfo[0] for float16 input expects float16 zloss/lseForZloss
    // SupportInfo[1] for float32 input expects float32 zloss/lseForZloss
    // SupportInfo[2] for bfloat16 input expects bfloat16 zloss/lseForZloss
    // So, we just use input.scalar_type() directly, as it covers all these cases.
    
    // zlossOut is always a scalar (shape [1])
    at::Tensor zlossOut = torch::empty({1}, input.options().dtype(zloss_output_dtype)); 
    
    // lseForZlossOut has shape [N, 1] for N samples, or [N] after squeeze in Python
    // aclnn signature implies [N, 1] if it comes from keepdim=True mean/max context
    // The provided error log shows lseForZlossOut with ND format, and you used input.size(0)
    // for its size in Python, so it's likely (batch_size,)
    at::Tensor lseForZlossOut = torch::empty({input.size(0)}, input.options().dtype(zloss_output_dtype));
    // --- END CHANGE ---

    // Prepare optional weight tensor for NPU_CMD
    at::Tensor weight_npu = weightOptional.has_value() ? weightOptional.value() : at::Tensor();

    const char* reduction_cstr = reductionOptional.c_str();
    EXEC_NPU_CMD(aclnnCrossEntropyLoss,
                 input, target, weight_npu, reduction_cstr,
                 ignoreIndex, labelSmoothing, lseSquareScaleForZloss, returnZloss,
                 lossOut, logProbOut, zlossOut, lseForZlossOut);

    std::vector<at::Tensor> result = {lossOut, logProbOut}; // lossOut and logProbOut are always returned by Model
    // Only append zloss related outputs if returnZloss is true
    if (returnZloss) {
        result.push_back(zlossOut);
        result.push_back(lseForZlossOut);
    }
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `cross_entropy_loss`
    m.def("cross_entropy_loss", &cross_entropy_loss, "");
}