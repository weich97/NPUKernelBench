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
 * - aclnnCustomOp -> aclnnReverseSequence
 * - custom_pybind_api -> reverse_sequence
 *
 * Implementation note.
 */
at::Tensor custom_pybind_api(at::Tensor x, at::Tensor seqLengths, int64_t seqDim, int64_t batchDim)
{
    // Implementation note.
    auto x_sizes = x.sizes().vec();
    auto seqLengths_sizes = seqLengths.sizes().vec();

    // Implementation note.
    TORCH_CHECK(seqLengths_sizes.size() == 1, "seqLengths must be a 1-D tensor.");
    TORCH_CHECK(seqLengths_sizes[0] == x_sizes[batchDim], "The size of seqLengths must be equal to the batch size.");
    TORCH_CHECK(seqDim >= 0 && seqDim < x_sizes.size(), "seqDim out of range");
    TORCH_CHECK(batchDim >= 0 && batchDim < x_sizes.size() && batchDim != seqDim, "batchDim out of range or equal to seqDim");

    // Implementation note.
    at::Tensor result = torch::empty_like(x);

    // Implementation note.
    EXEC_NPU_CMD(aclnnReverseSequence, x, seqLengths, seqDim, batchDim, result);

    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("reverse_sequence", &custom_pybind_api, "");
}