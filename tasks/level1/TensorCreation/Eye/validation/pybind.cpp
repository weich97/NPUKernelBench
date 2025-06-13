#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &input, int64_t num_rows, int64_t num_columns,
                            std::vector<int64_t> batch_shape, int64_t dtype)
{
    aclIntArray* batch_shape_acl = ConvertType(at::IntArrayRef(batch_shape));
    EXEC_NPU_CMD(aclnnCustomOp, input, num_rows, num_columns, batch_shape_acl, dtype);
    return input;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}