import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        return torch.cos(x)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.cos_math(x)