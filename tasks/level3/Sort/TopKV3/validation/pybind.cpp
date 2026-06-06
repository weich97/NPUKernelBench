#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <tuple>
#include <vector> // Implementation note.

/**
 * Implementation note.
 * Implementation note.
 * Implementation note.
 */
std::vector<at::Tensor> custom_pybind_api(
    at::Tensor &self_tensor, 
    int64_t k, 
    int64_t dim, 
    bool largest, 
    bool sorted
) {
    // Implementation note.
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < self_tensor.dim(); ++i) {
        if (i == dim) {
            output_shape.push_back(k); // Implementation note.
        } else {
            output_shape.push_back(self_tensor.size(i)); // Implementation note.
        }
    }

    // Implementation note.
    at::Tensor values = torch::empty(output_shape, self_tensor.options().dtype(torch::kFloat16));
    at::Tensor indices = torch::empty(output_shape, self_tensor.options().dtype(torch::kInt64));

    EXEC_NPU_CMD(aclnnCustomOp, self_tensor, k, dim, largest, sorted, values, indices);
    
    return {values, indices};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}