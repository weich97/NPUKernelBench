// pybind.py
/**
 * @file extension_group_norm_swish_grad.cpp
 */
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
// This is the actual C++ function that calls aclnn
// It takes outputs as non-optional pointers (aclTensor*) or optional C++ types
// Based on the error, it's safer to provide non-null pointers always
void _group_norm_swish_grad_internal( // Changed name to internal for clarity
    at::Tensor dy,
    at::Tensor mean,
    at::Tensor rstd,
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    int64_t group,
    std::string dataFormat,
    double swishScale,
    bool dgammaIsRequire,
    bool dbetaIsRequire,
    at::Tensor dxOut,
    at::Tensor dgammaOut,
    at::Tensor dbetaOut)
{
    // Prepare string for NPU_CMD
    const char* data_format_cstr = dataFormat.c_str();

    // Call the NPU operator
    // Now, dgammaOut and dbetaOut are always at::Tensor, caller ensures they are valid.
    EXEC_NPU_CMD(aclnnGroupNormSwishGrad,
                 dy, mean, rstd, x, gamma, beta,
                 group, data_format_cstr, swishScale,
                 dgammaIsRequire, dbetaIsRequire,
                 dxOut, dgammaOut, dbetaOut); // Pass non-optional tensors
}


// This is the wrapper function exposed to Python
std::vector<at::Tensor> group_norm_swish_grad_wrapper(
    at::Tensor dy,
    at::Tensor mean,
    at::Tensor rstd,
    at::Tensor x,
    at::Tensor gamma,
    at::Tensor beta,
    int64_t group,
    std::string dataFormat,
    double swishScale,
    bool dgammaIsRequire,
    bool dbetaIsRequire)
{
    at::Tensor dxOut = torch::empty_like(x);
    
    // --- MODIFICATION START ---
    // Always create dgammaOut and dbetaOut, even if not strictly required
    // If not required, create a zero-filled tensor of the correct shape.
    // This ensures the pointer passed to the internal NPU kernel is never nullptr.
    at::Tensor dgammaOut_actual;
    if (dgammaIsRequire) {
        dgammaOut_actual = torch::empty_like(gamma);
    } else {
        dgammaOut_actual = torch::zeros_like(gamma); // Create a zero tensor if not required
    }

    at::Tensor dbetaOut_actual;
    if (dbetaIsRequire) {
        dbetaOut_actual = torch::empty_like(beta);
    } else {
        dbetaOut_actual = torch::zeros_like(beta); // Create a zero tensor if not required
    }
    // --- MODIFICATION END ---

    // Call the internal C++ function which does the actual NPU call
    _group_norm_swish_grad_internal(
        dy, mean, rstd, x, gamma, beta,
        group, dataFormat, swishScale,
        dgammaIsRequire, dbetaIsRequire,
        dxOut, dgammaOut_actual, dbetaOut_actual // Pass these non-optional tensors
    );

    std::vector<at::Tensor> result;
    result.push_back(dxOut);
    
    // Only push required outputs to the Python list that will be returned
    if (dgammaIsRequire) {
        result.push_back(dgammaOut_actual);
    }
    
    if (dbetaIsRequire) {
        result.push_back(dbetaOut_actual);
    }

    return result;
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("group_norm_swish_grad", &group_norm_swish_grad_wrapper, "");
}