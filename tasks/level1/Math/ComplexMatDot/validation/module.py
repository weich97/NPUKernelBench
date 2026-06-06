from typing import List

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

    def forward(self, matx: torch.Tensor, maty: torch.Tensor, m: int, n: int) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        return matx * maty


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, matx: torch.Tensor, maty: torch.Tensor, m: int, n: int) -> torch.Tensor:
        return kernel_gen_ops.complex_mat_dot(matx, maty, m, n)