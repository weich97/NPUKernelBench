from typing import List, Optional, Tuple
import torch
import torch_npu # Assuming NPU environment
from torch_npu.contrib import transfer_to_npu # Assuming NPU environment
import torch.nn as nn
import kernel_gen_ops # Custom ops binding
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], scalar_weight: float) -> List[torch.Tensor]:
        """
        Native PyTorch implementation of ForeachLerpScalar.
        Performs y_i = x1_i + weight * (x2_i - x1_i) for each tensor in the lists.
        """
        if not (isinstance(x1, list) and isinstance(x2, list)):
            raise TypeError("Inputs x1 and x2 must be lists of tensors.")
        if len(x1) != len(x2):
            raise ValueError("Input tensor lists x1 and x2 must have the same length.")

        output_list = []
        for i in range(len(x1)):
            # torch.lerp(input, end, weight) -> input + weight * (end - input)
            # This directly maps to x1_i + weight * (x2_i - x1_i)
            # F.lerp also works the same way
            result_tensor = torch.lerp(x1[i], x2[i], scalar_weight)
            output_list.append(result_tensor)
        return output_list

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], scalar_weight: float) -> List[torch.Tensor]:
        """
        Custom operator implementation for ForeachLerpScalar.
        Calls the kernel_gen_ops.foreach_lerp_scalar.
        """
        # Pass the scalar value directly to the custom op if it expects a scalar float/int,
        # or pass the tensor if the C++ op expects a scalar tensor.
        # Assuming the C++ op expects a torch.Tensor for scalar weight.
        return kernel_gen_ops.foreach_lerp_scalar(x1, x2, torch.tensor(scalar_weight, dtype=torch.float32))

