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
std::vector<at::Tensor> custom_pybind_api(std::vector<at::Tensor> &x, at::Tensor &scalar)
{
    // alloc output memory
    std::vector<at::Tensor> result;
    result.reserve(x.size());

    for (const auto &tensor : x) {
        at::Tensor scalar_tensor = at::empty({}, tensor.options());
        result.push_back(scalar_tensor);
    }

    // Convert std::vector<at::Tensor> to at::TensorList for the EXEC_NPU_CMD
    at::TensorList result_list = at::TensorList(result);
    at::TensorList input = at::TensorList(x);
   
    auto scalarValue = ConvertTensorToAclScaler(scalar);
    EXEC_NPU_CMD(aclnnCustomOp, input, scalarValue, result_list);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}