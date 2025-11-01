#include <torch/extension.h>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> foreach_pow_scalar_and_tensor_wrapper(
    std::vector<at::Tensor> &x,
    at::Tensor scalar) {

    std::vector<at::Tensor> result;
    result.reserve(x.size());

    for (const auto& t : x) {
        result.emplace_back(torch::zeros_like(t));
    }

    // Convert std::vector<at::Tensor> to at::TensorList for the EXEC_NPU_CMD
    at::TensorList result_list = at::TensorList(result);
    at::TensorList x_list = at::TensorList(x);

    // Host 侧标量
    auto base_scalar = ConvertTensorToAclScaler(scalar);
    //c10::scalar_base scalar(scalar.item<float>());

    EXEC_NPU_CMD(aclnnForeachPowScalarAndTensor,
                 base_scalar,   // aclScalar*
                 x_list,        // aclTensorList*
                 result_list);       // aclTensorList*

    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.def("foreach_pow_scalar_and_tensor", &foreach_pow_scalar_and_tensor_wrapper, "");
}