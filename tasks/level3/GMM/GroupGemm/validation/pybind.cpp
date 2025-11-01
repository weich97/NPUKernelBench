#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_fun(at::Tensor &a, at::Tensor &b, at::Tensor &c, at::Tensor & alpha, at::Tensor & beta,
                            at::IntArrayRef &m_list, at::IntArrayRef &k_list, at::IntArrayRef &n_list)
{
    aclIntArray* m_list_arr = ConvertType(m_list);
    aclIntArray* k_list_arr = ConvertType(k_list);
    aclIntArray* n_list_arr = ConvertType(n_list);
    EXEC_NPU_CMD(aclnnCustomOp, a, b, c, alpha, beta, m_list_arr, k_list_arr, n_list_arr);
    return c;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_fun, "");
}