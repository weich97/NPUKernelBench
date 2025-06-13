#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */
std::vector<at::Tensor> apply_adam_wv2(at::Tensor &var_ref, at::Tensor &m_ref, at::Tensor &v_ref, at::Tensor &grad,
                                       at::Tensor &step, at::Tensor max_grad_norm_ref,
                                       float lr, float beta1, float beta2,
                                       float weight_decay, float eps, bool amsgrad, bool maximize)
{
     EXEC_NPU_CMD(aclnnApplyAdamWV2, var_ref, m_ref, v_ref, grad, step, max_grad_norm_ref,
                    lr, beta1, beta2, weight_decay, eps, amsgrad, maximize);

    std::vector<at::Tensor> result;
    // 根据输入准备输出张量
    if (amsgrad) {
        result = {var_ref, m_ref, v_ref, max_grad_norm_ref};
    } else {
        result = {var_ref, m_ref, v_ref};
    }
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("apply_adam_wv2", &apply_adam_wv2, "ApplyAdamWV2 optimizer operation");
}