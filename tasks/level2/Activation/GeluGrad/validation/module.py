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

    def forward(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        orig_dtype = x.dtype
        x = x.float()
        dy = dy.float()
        
        # Implementation note.
        c0 = -0.0713548162726002527220
        c1 = -1.595769121605730711759
        c2 = 0.2140644488178007
        c3 = 1.595769121605730711759

        x_square = x * x
        px_arg = (x_square * c0 + c1) * x
        px = torch.exp(px_arg)
        res0 = (x_square * c2 + c3) * x
        t_denominator = px + 1.0
        t = 1.0 / t_denominator
        resp_intermediate = px * res0 * t.pow(2) + t
        resp = torch.nan_to_num(resp_intermediate, nan=0.0)
        z = dy * resp

        return z.to(orig_dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        import kernel_gen_ops
        # Implementation note.
        y = torch.nn.functional.gelu(x)

        # Implementation note.
        return kernel_gen_ops.gelu_grad(dy, x, y)