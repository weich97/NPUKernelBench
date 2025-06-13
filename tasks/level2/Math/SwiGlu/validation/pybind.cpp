#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 * 使用说明：
 * 1. 将此文件中的 aclnnCustomOp 替换为实际算子名称，如 aclnnForeachExp
 * 2. 将 custom_pybind_api 替换为对应的下划线命名形式，如 foreach_exp
 *
 * 替换示例：
 * - aclnnCustomOp -> aclnnXXX (其中XXX为算子名，如替换为aclnnForeachExp)
 * - custom_pybind_api -> YYY (其中YYY为算子名的下划线形式，如foreach_exp)
 *
 * 注意：替换时需保持函数签名和逻辑不变，仅修改上述指定的名称，这一替换过程将在batch_compile.py文件中自动被执行
 */
 at::Tensor custom_pybind_api(at::Tensor x)
 {
    
     // 获取输入张量的形状
     auto x_sizes = x.sizes().vec();  // 转为 std::vector<int64_t>
     
     // 将最后一维减半
     TORCH_CHECK(x_sizes.back() % 2 == 0, "Last dimension must be divisible by 2 for SwiGLU.");
     x_sizes.back() /= 2;
 
     // 创建 result 张量：与 x 类型一致，形状为后一半大小
     at::Tensor result = torch::empty(x_sizes, x.options());
 
     int dimoptional = -1;

     // 执行 NPU 自定义算子
     EXEC_NPU_CMD(aclnnCustomOp, x, dimoptional, result);
 
     return result;
 }
 
 
PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}