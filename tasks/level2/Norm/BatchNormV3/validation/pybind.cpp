/**
 * @file extension_batch_norm_v3.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp" // Ensure this includes ConvertType and EXEC_NPU_CMD

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
std::vector<at::Tensor> batch_norm_v3(
    at::Tensor input,
    c10::optional<at::Tensor> weightOptional,
    c10::optional<at::Tensor> biasOptional,
    at::Tensor runningMean,
    at::Tensor runningVar,
    bool training,
    double momentum,
    double eps)
{
    // Output tensors: output, saveMean, saveInvstd
    at::Tensor output = torch::empty_like(input);
    
    // saveMean and saveInvstd have the same shape as runningMean/runningVar
    at::Tensor saveMean = torch::empty_like(runningMean);
    at::Tensor saveInvstd = torch::empty_like(runningVar);

    // Prepare optional weight and bias tensors for NPU_CMD
    at::Tensor weight_npu = weightOptional.has_value() ? weightOptional.value() : at::Tensor();
    at::Tensor bias_npu = biasOptional.has_value() ? biasOptional.value() : at::Tensor();

    EXEC_NPU_CMD(aclnnBatchNorm,
                 input, weight_npu, bias_npu,
                 runningMean, runningVar, // These are modified in-place by NPU op
                 training, momentum, eps,
                 output, saveMean, saveInvstd);

    std::vector<at::Tensor> result = {output, saveMean, saveInvstd};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `batch_norm_v3`
    m.def("batch_norm_v3", &batch_norm_v3, "");
}