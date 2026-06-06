#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &add_0_input0, at::Tensor &add_0_input1, at::Tensor &mult_0_input1, at::Tensor &mult_1_input1, at::Tensor &mult_2_input1)
{
        
    // Implementation note.
    int64_t dim0 = add_0_input0.size(0);
    int64_t dim2 = add_0_input0.size(2);

    // Implementation note.
    at::Tensor result = torch::empty({dim0, dim2}, add_0_input0.options());

    EXEC_NPU_CMD(aclnnCustomOp, add_0_input0, add_0_input1, mult_0_input1,mult_1_input1, mult_2_input1, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}