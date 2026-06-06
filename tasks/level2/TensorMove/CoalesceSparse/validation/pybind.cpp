#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp" // Assuming this contains EXEC_NPU_CMD macro and necessary headers for ACLNN

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
std::vector<at::Tensor> custom_pybind_api(at::Tensor uniqueLen, at::Tensor uniqueIndices,
                               at::Tensor indices, at::Tensor values)
{
    int64_t new_nnz = uniqueLen.numel();
    int64_t sparse_dim = indices.size(1);

    at::Tensor newIndices = torch::empty({new_nnz, sparse_dim}, indices.options());

    std::vector<int64_t> new_values_shape_vec;
    new_values_shape_vec.push_back(new_nnz);
    for (int i = 1; i < values.dim(); ++i) {
        new_values_shape_vec.push_back(values.size(i));
    }
    at::Tensor newValues = torch::empty(new_values_shape_vec, values.options());
    newValues.zero_(); 
    EXEC_NPU_CMD(aclnnCoalesceSparse, uniqueLen, uniqueIndices, indices, values, newIndices, newValues);
    std::vector<at::Tensor> result_tensors;
    result_tensors.push_back(newIndices);
    result_tensors.push_back(newValues);
    return result_tensors;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("coalesce_sparse", &custom_pybind_api, "");
}