#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor custom_pybind_api(std::vector<int64_t> &dims, at::Tensor &value) {
    if (value.numel() != 1) {
        throw std::runtime_error("Input tensor must be a scalar (have exactly one element)");
    }

    at::Tensor result = torch::zeros(dims, value.options());

    aclIntArray* dims_acl = ConvertType(at::IntArrayRef(dims));

    EXEC_NPU_CMD(aclnnCustomOp, dims_acl, value, result);

    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}