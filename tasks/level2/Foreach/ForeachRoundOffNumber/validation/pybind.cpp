/**
 * @file extension_foreach_round_off_number.cpp
 */
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
std::vector<at::Tensor> foreach_round_off_number(std::vector<at::Tensor> &x,
                                                 at::Tensor &round_mode_tensor) {
    // Implementation note.
    std::vector<at::Tensor> result;
    result.reserve(x.size());
    for (const auto& t : x) {
        result.emplace_back(torch::empty_like(t));
    }

    // Implementation note.
    at::TensorList result_list = at::TensorList(result);
    at::TensorList input_list  = at::TensorList(x);

    // Implementation note.
    EXEC_NPU_CMD(aclnnForeachRoundOffNumber,
                 input_list,
                 round_mode_tensor,
                 result_list);

    // Implementation note.
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `foreach_round_off_number`
    m.def("foreach_round_off_number", &foreach_round_off_number, "");
}