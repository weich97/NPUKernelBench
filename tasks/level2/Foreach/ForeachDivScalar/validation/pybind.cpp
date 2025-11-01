#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

std::vector<at::Tensor> custom_pybind_api(std::vector<at::Tensor> &x, at::Tensor &scalar)
{
    // alloc output memory
    std::vector<at::Tensor> result;
    result.reserve(x.size());

    for (const auto &tensor : x) {
        result.push_back(torch::empty_like(tensor));
    }

    // Convert std::vector<at::Tensor> to at::TensorList for the EXEC_NPU_CMD
    at::TensorList result_list = at::TensorList(result);
    at::TensorList input = at::TensorList(x);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnCustomOp, input, scalar, result_list);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}