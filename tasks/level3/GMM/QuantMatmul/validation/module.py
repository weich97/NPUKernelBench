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

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, per_token_scale: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """

        accumulator = torch.matmul(a.to(torch.int32), b.to(torch.int32))
        output = accumulator.to(torch.float32) * scale.to(torch.float32).unsqueeze(0) * per_token_scale.to(torch.float32).unsqueeze(1)
        return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, per_token_scale: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.quant_matmul(a, b, scale, per_token_scale)