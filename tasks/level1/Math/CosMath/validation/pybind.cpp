#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

/**
 * 使用说明：
 * 此文件为Cos算子的pybind接口文件，提供了custom_pybind_api函数，用于调用NPU自定义算子进行计算。
 * 输入参数为一个at::Tensor类型的张量，输出为一个at::Tensor类型的张量，该张量是输入张量每个元素的余弦值。
 * 在调用时，首先会分配输出内存，然后调用EXEC_NPU_CMD来执行NPU自定义算子进行计算。
 */
at::Tensor custom_pybind_api(at::Tensor &x)
{
    // alloc output memory
    at::Tensor result = torch::empty_like(x);

    // call aclnn interface to perform the computation
    EXEC_NPU_CMD(aclnnCustomOp, x, result);
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}