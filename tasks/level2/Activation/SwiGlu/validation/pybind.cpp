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
 at::Tensor custom_pybind_api(at::Tensor x, int64_t dim = -1)
 {
    
     // Implementation note.
     auto x_sizes = x.sizes().vec(); // Implementation note.
     
     // Implementation note.
     int64_t actual_dim = dim < 0 ? dim + x.dim() : dim;
     TORCH_CHECK(actual_dim >= 0 && actual_dim < x.dim(),
                 "dim must be in range [", -x.dim(), ", ", x.dim()-1, "]");

     TORCH_CHECK(x_sizes[actual_dim] % 2 == 0,
                 "Dimension ", actual_dim, " must be divisible by 2 for SwiGLU.");
                 
     x_sizes[actual_dim] /= 2;
 
     // Implementation note.
     at::Tensor result = torch::empty(x_sizes, x.options());

     // Implementation note.
     EXEC_NPU_CMD(aclnnCustomOp, x, actual_dim, result);
 
     return result;
 }
 
 
PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}