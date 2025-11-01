#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
std::vector<at::Tensor> custom_pybind_api(std::vector<at::Tensor> &varRef, at::Tensor &indice, at::Tensor &updates, c10::optional<at::Tensor> &mask, const char* reduce, int64_t axis)
{
    at::TensorList varRefList = at::TensorList(varRef);
    at::Tensor mask_npu = mask.has_value() ? mask.value() : at::Tensor();
    EXEC_NPU_CMD(aclnnCustomOp, varRefList, indice, updates, mask_npu, reduce, axis);
    return varRef;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}