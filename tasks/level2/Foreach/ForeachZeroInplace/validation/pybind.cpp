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
std::vector<at::Tensor> custom_pybind_api(std::vector<at::Tensor> &x)
{
    // alloc output memory
    std::vector<at::Tensor> result;
    result.reserve(x.size());

    for (const auto &tensor : x) {
        result.push_back(torch::empty_like(tensor));
    }

    // Convert std::vector<at::Tensor> to at::TensorList for the EXEC_NPU_CMD
    at::TensorList input = at::TensorList(x);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnCustomOp, input);

    // Convert at::TensorList back to std::vector<at::Tensor>
    std::vector<at::Tensor> output;
    output.reserve(input.size());
    for (const auto &tensor : input) {
        output.push_back(tensor);
    }

    return output;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}