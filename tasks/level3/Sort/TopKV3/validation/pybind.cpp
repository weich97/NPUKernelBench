#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"
#include <tuple>
#include <vector>  // 需要包含此头文件以使用 std::vector

/**
 * 注册 NPU 设备的前向实现
 * 说明：此文件是 TopKV3 算子的 pybind 接口文件，custom_pybind_api 函数封装了 NPU 自定义算子的执行。
 * 函数的参数应与 module.py 中 ModelNew 的 forward 方法参数一致，并通过 EXEC_NPU_CMD 执行 NPU 自定义算子。
 */
std::vector<at::Tensor> custom_pybind_api(
    at::Tensor &self_tensor, 
    int64_t k, 
    int64_t dim, 
    bool largest, 
    bool sorted
) {
    // 动态计算输出形状
    std::vector<int64_t> output_shape;
    for (int64_t i = 0; i < self_tensor.dim(); ++i) {
        if (i == dim) {
            output_shape.push_back(k);  // 在目标维度上取 k 个值
        } else {
            output_shape.push_back(self_tensor.size(i));  // 其他维度保持不变
        }
    }

    // 创建输出张量
    at::Tensor values = torch::empty(output_shape, self_tensor.options().dtype(torch::kFloat16));
    at::Tensor indices = torch::empty(output_shape, self_tensor.options().dtype(torch::kInt64));

    EXEC_NPU_CMD(aclnnCustomOp, self_tensor, k, dim, largest, sorted, values, indices);
    
    return {values, indices};
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}