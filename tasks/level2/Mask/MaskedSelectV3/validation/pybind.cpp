#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &input_tensor, at::Tensor &mask)
{
    // 1. 检查输入形状是否可广播
    auto broadcast_shape = at::infer_size(input_tensor.sizes(), mask.sizes());
    int64_t broadcast_size = 1;
    for (auto dim : broadcast_shape) broadcast_size *= dim;

    // 2. 创建临时输出张量（1D，大小 = broadcast_size）
    at::Tensor tmp_output = torch::empty({broadcast_size}, input_tensor.options());

    // 3. 调用 NPU 算子
    EXEC_NPU_CMD(aclnnCustomOp, input_tensor, mask, tmp_output);

    // 4. 截取实际选中的部分
    int64_t selected_size = mask.sum().item<int64_t>();
    return tmp_output.narrow(0, 0, selected_size);
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}