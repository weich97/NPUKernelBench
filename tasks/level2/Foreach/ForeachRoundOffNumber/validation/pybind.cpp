/**
 * @file extension_foreach_round_off_number.cpp
 */
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * 使用说明：
 * 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnForeachExp
 * 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 foreach_exp
 *
 * 替换示例：
 * - aclnnCustomOp -> aclnnXXX (其中XXX为算子名，如替换为aclnnForeachExp)
 * - custom_pybind_api -> YYY (其中YYY为算子名的下划线形式，如foreach_exp)
 *
 * 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
 */
std::vector<at::Tensor> foreach_round_off_number(std::vector<at::Tensor> &x,
                                                 at::Tensor &round_mode_tensor) {
    /* 1. 拥有者：std::vector */
    std::vector<at::Tensor> result;
    result.reserve(x.size());
    for (const auto& t : x) {
        result.emplace_back(torch::empty_like(t));
    }

    /* 2. 视图：at::TensorList */
    at::TensorList result_list = at::TensorList(result);
    at::TensorList input_list  = at::TensorList(x);

    /* 4. 调用 ACLNN */
    EXEC_NPU_CMD(aclnnForeachRoundOffNumber,
                 input_list,
                 round_mode_tensor,
                 result_list);

    /* 5. 返回拥有者 */
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    // Function name in Python will be `foreach_round_off_number`
    m.def("foreach_round_off_number", &foreach_round_off_number, "");
}