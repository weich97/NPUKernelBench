#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor custom_pybind_api(at::Tensor &x1, at::Tensor &x2, double t1, double t2, double t3)
{
    // 计算输出形状: [x1的batch维, x2的剩余维]
    std::vector<int64_t> output_shape;
    output_shape.push_back(x1.size(0)); // batch维
    for (int i = 1; i < x2.dim(); i++) {
        output_shape.push_back(x2.size(i)); // x2的剩余维
    }

    // 分配正确形状的输出内存
    at::Tensor result = torch::empty(output_shape, x1.options());

    // 调用ACLNN接口执行计算
    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, t1, t2, t3, result);
    
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}