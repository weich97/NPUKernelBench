#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * register forward implementation for NPU device
 */

at::Tensor custom_pybind_api(at::Tensor &start, at::Tensor &end, at::Tensor &step)
{
    auto scalarStart = ConvertTensorToAclScaler(start.cpu());
    auto scalarEnd = ConvertTensorToAclScaler(end);
    auto scalarStep = ConvertTensorToAclScaler(step);

    double value_start = start.to(torch::kFloat64).item<double>();
    double value_end = end.to(torch::kFloat64).item<double>();
    double value_step = step.to(torch::kFloat64).item<double>();

    double size_arange = ceil(static_cast<double>(value_end - value_start) / value_step);
    int64_t size_value = static_cast<int64_t>(size_arange);

    // c10::DeviceIndex indexFromCurDevice = c10_npu::current_device();
    // at::Device device = at::Device(c10::DeviceType::PrivateUse1, indexFromCurDevice);
    at::Tensor result = torch::empty(size_value, start.options());

    EXEC_NPU_CMD(aclnnCustomOp, scalarStart, scalarEnd, scalarStep, result);

    return result;
}

PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("custom_pybind_api", &custom_pybind_api, "");
}







