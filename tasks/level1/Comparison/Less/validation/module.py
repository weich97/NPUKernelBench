import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """Reference PyTorch model for the Less operator."""

    def __init__(self):
        """Initialize the reference model."""
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Execute element-wise less-than comparison.

        Args:
            x1: First input tensor.
            x2: Second input tensor.

        Returns:
            Element-wise comparison result tensor.
        """
        return torch.less(x1, x2)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.less(x1, x2)
