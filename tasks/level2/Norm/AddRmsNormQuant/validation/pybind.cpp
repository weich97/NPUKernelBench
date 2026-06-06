/**
 * @file extension_add.cpp
 */
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
std::vector<at::Tensor> custom_pybind_api(
    at::Tensor x1, 
    at::Tensor x2, 
    at::Tensor gamma, 
    at::Tensor scales1, 
    c10::optional<at::Tensor> scales2, 
    c10::optional<at::Tensor> zero_points1,
    c10::optional<at::Tensor> zero_points2,
    int64_t axis = -1,
    float epsilon = 1e-5f, 
    bool div_mode = true )
{
    // Implementation note.
    torch::IntArrayRef shapes = x1.sizes();
    std::vector<int64_t> meanOutShape(shapes.begin(), shapes.end());
    for(int i = 0; i < gamma.sizes().size(); i++) {
        meanOutShape[meanOutShape.size() - i - 1] = 1;
    }
    auto options = torch::TensorOptions().dtype(torch::kInt8).device(x1.device()); // rstd is usually float
    at::Tensor yOut1 = torch::empty_like(x1, options);
    at::Tensor yOut2 = torch::empty_like(x1, options);

    at::Tensor xOut = torch::empty_like(x1);
   
    EXEC_NPU_CMD(aclnnCustomOp, x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, axis, epsilon, div_mode, yOut1, yOut2, xOut);
    
    std::vector<at::Tensor> result;
    result = {yOut1};
    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}