#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

std::vector<at::Tensor> custom_pybind_api(std::vector<at::Tensor> &x1, std::vector<at::Tensor> &x2, std::vector<at::Tensor> &x3, at::Tensor &scalars)
{
    // alloc output memory
    std::vector<at::Tensor> result;
    result.reserve(x1.size());

    for (const auto &tensor : x1) {
        result.push_back(torch::empty_like(tensor));
    }

    // Convert std::vector<at::Tensor> to at::TensorList for the EXEC_NPU_CMD
    at::TensorList result_list = at::TensorList(result);
    at::TensorList input1 = at::TensorList(x1);
    at::TensorList input2 = at::TensorList(x2);
    at::TensorList input3 = at::TensorList(x3);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnCustomOp, input1, input2, input3, scalars, result_list);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}