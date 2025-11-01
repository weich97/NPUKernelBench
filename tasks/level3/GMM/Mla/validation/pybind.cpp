#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_fun(
    at::Tensor &query_nope,
    at::Tensor &query_rope,
    at::Tensor &kv_nope_cache,
    at::Tensor &kv_rope_cache,
    at::Tensor &block_tables,
    at::IntArrayRef &q_seqlen_list,
    at::IntArrayRef &k_seqlen_list)
{
    // Calculate output tensor shape: [num_tokens, num_heads, head_size]
    int64_t num_tokens = query_nope.size(0);
    int64_t num_heads = query_nope.size(1);
    int64_t head_size = kv_nope_cache.size(3);
    at::Tensor result = torch::empty({num_tokens, num_heads, head_size}, query_nope.options());

    // Execute the custom NPU kernel
    EXEC_NPU_CMD(aclnnCustomOp, query_nope, query_rope, kv_nope_cache, kv_rope_cache, block_tables, q_seqlen_list, k_seqlen_list, result);

    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_fun, "");
}