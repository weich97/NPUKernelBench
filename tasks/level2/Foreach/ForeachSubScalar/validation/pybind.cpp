#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"


std::vector<at::Tensor> custom_pybind_api(std::vector<at::Tensor> &x, at::Tensor scalar_tensor)
{
    // 其余代码不变
    std::vector<at::Tensor> result;
    result.reserve(x.size());

    for (const auto &tensor : x) {
        result.push_back(torch::empty_like(tensor));
    }

    at::TensorList result_list = at::TensorList(result);
    at::TensorList input = at::TensorList(x);

    EXEC_NPU_CMD(aclnnCustomOp, input, scalar_tensor, result_list);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}