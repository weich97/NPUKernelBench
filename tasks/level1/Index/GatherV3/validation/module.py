from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import math


class Model(nn.Module):
    """Reference PyTorch model for the GatherV3 operator."""

    def __init__(self):
        """Initialize the reference model."""
        super(Model, self).__init__()

    def forward(self, self_tensor: torch.Tensor, axis: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        Execute the GatherV3/index_select operation.

        Args:
            self_tensor: Source input tensor.
            indices: Index tensor.
            axis: Axis tensor.

        Returns:
            GatherV3 output tensor.
        """
        output = torch.index_select(self_tensor, dim=axis, index=indices)
        return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, self_tensor: torch.Tensor, axis: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.gather_v3(self_tensor, axis, indices)
