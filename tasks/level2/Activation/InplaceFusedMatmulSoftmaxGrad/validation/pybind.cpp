#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

//用cann_8.1.RC1可以通过所有测试；用8.0.RC3.20会报出"no template named 'Broadcast' in namespace 'AscendC'; did you mean 'BroadCast'?"错

/**
 * register forward implementation for NPU device
 */
at::Tensor custom_pybind_api(at::Tensor &softmaxOutput, at::Tensor &gradOutput, at::Tensor &values)
{   
    EXEC_NPU_CMD(aclnnCustomOp, softmaxOutput, gradOutput, values);
    return softmaxOutput;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}