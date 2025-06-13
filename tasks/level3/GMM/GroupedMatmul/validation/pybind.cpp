#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>
#include "pytorch_npu_helper.hpp"

/**
 * Convert single tensor to tensor list if needed
 */
std::vector<at::Tensor> ConvertToTensorVector(const c10::IValue& input) {
    if (input.isTensor()) {
        return {input.toTensor()};
    } else if (input.isTensorList()) {
        return input.toTensorList().vec();
    }
    TORCH_CHECK(false, "Input must be Tensor or List[Tensor]");
}

/**
 * register forward implementation for NPU device
 */
c10::IValue grouped_matmul_custom(
    const c10::IValue& x,
    const std::vector<at::Tensor>& weight,
    const c10::optional<std::vector<at::Tensor>>& bias,
    const c10::optional<std::vector<int64_t>>& group_list,
    int64_t split_item,
    bool transpose_weight,
    bool transpose_x)
{
    // Convert x to tensor vector
    auto x_vec = ConvertToTensorVector(x);

    // Prepare group_list as IntArrayRef
    at::IntArrayRef group_list_ref;
    if (group_list.has_value()) {
        group_list_ref = at::IntArrayRef(*group_list);
    } else {
        group_list_ref = at::IntArrayRef();  // Empty IntArrayRef
    }

    // Calculate output shapes
    std::vector<std::vector<int64_t>> output_shapes;
    int64_t total_m = 0;

    for (size_t i = 0; i < weight.size(); ++i) {
        int64_t m, n;

        // Calculate m dimension
        if (x_vec.size() == 1) {
            // Single x tensor case
            if (group_list.has_value() && !group_list->empty()) {
                int64_t start = (i == 0) ? 0 : (*group_list)[i-1];
                int64_t end = (*group_list)[i];
                m = end - start;
            } else {
                m = x_vec[0].size(transpose_x ? 1 : 0);
            }
        } else {
            // Multiple x tensors case
            m = x_vec[i].size(transpose_x ? 1 : 0);
        }

        // Calculate n dimension
        n = weight[i].size(transpose_weight ? 0 : 1);

        output_shapes.push_back({m, n});
        total_m += m;
    }

    // Allocate output tensors based on split_item
    std::vector<at::Tensor> outputs;

    if (split_item == 2 || split_item == 3) {
        // Single output tensor case - concatenated shape
        // Assume all n dimensions are the same
        int64_t n = output_shapes[0][1];
        outputs.push_back(at::empty({total_m, n}, x_vec[0].options()));
    } else {
        // Multiple output tensors case
        for (const auto& shape : output_shapes) {
            outputs.push_back(at::empty(shape, x_vec[0].options()));
        }
    }

    // Call the operator
    EXEC_NPU_CMD(aclnnGroupedMatmul,
                 x,
                 weight,
                 bias,
                 group_list_ref,
                 split_item,
                 transpose_weight,
                 transpose_x,
                 outputs);

    // Return based on split_item
    if (split_item == 2 || split_item == 3) {
        // Single output tensor case
        return outputs[0];
    } else {
        // Multiple output tensors case
        return c10::IValue(outputs);
    }
}


PYBIND11_MODULE(kernel_gen_ops, m) {
    m.doc() = "Python bindings for kernel_gen_ops";
    m.def("grouped_matmul", &grouped_matmul_custom,
          "Grouped matrix multiplication",
          py::arg("x"),
          py::arg("weight"),
          py::arg("bias") = py::none(),
          py::arg("group_list") = py::none(),
          py::arg("split_item") = 0,
          py::arg("transpose_weight") = false,
          py::arg("transpose_x") = false);
}