#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

std::vector<at::Tensor> foreach_non_finite_check_and_unscale_wrapper(
    std::vector<at::Tensor> scaled_grads,
    at::Tensor found_inf_tensor_io,
    at::Tensor in_scale_tensor) {

    at::TensorList scaled_grads_list = at::TensorList(scaled_grads);

    EXEC_NPU_CMD(aclnnForeachNonFiniteCheckAndUnscale,
                 scaled_grads_list,
                 found_inf_tensor_io,
                 in_scale_tensor);

    std::vector<at::Tensor> ret;
    ret.reserve(scaled_grads_list.size() + 1);
    for (const auto& g : scaled_grads_list) ret.emplace_back(g);
    ret.emplace_back(found_inf_tensor_io);
    return ret;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.def("foreach_non_finite_check_and_unscale",
          &foreach_non_finite_check_and_unscale_wrapper, "");
}