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

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        # Implementation note.
        return torch.nn.functional.gelu(input_tensor)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        import kernel_gen_ops
        return kernel_gen_ops.gelu(input_tensor)