#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &query, at::Tensor &key, at::Tensor &value, double scale_value, int64_t head_num)
{
    const int64_t batch = query.size(0);   // 1
    const int64_t seq_len = query.size(1);   // 2048
    const int64_t hidden_dim = query.size(2);

    at::Tensor softmax_max = torch::empty({batch, head_num, seq_len, 8}, query.options().dtype(torch::kFloat32));
    at::Tensor softmax_sum = torch::empty({batch, head_num, seq_len, 8}, query.options().dtype(torch::kFloat32));
    at::Tensor attention_out = torch::empty({batch, seq_len, hidden_dim}, query.options());

    EXEC_NPU_CMD(aclnnCustomOp, query, key, value, scale_value, head_num, softmax_max, softmax_sum, attention_out);
    return attention_out;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}