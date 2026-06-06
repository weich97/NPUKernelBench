from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: torch.float32, beta: torch.float32) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        output = alpha * (a @ x) + beta * y
        return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: torch.float32, beta: torch.float32) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.gemv(a, x, y, alpha, beta)