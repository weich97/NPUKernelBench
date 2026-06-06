#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &feeds, at::Tensor &feeds_repeat_times, int64_t output_feeds_size)
{
    // Implementation note.
    auto output_size = feeds.sizes().vec();
    output_size[0] = output_feeds_size;
    at::Tensor result = torch::empty(output_size, feeds.options());

    // Implementation note.
    EXEC_NPU_CMD(aclnnCustomOp, feeds, feeds_repeat_times, output_feeds_size, result);
    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}