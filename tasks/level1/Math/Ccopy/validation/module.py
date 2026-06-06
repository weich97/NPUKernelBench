from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import math

class Model(nn.Module):
    def __init__(self, n: int, incx: int, incy: int):
        super(Model, self).__init__()
        self.n = n
        self.incx = incx
        self.incy = incy
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Implementation note.
        x_indices = torch.arange(self.n, device=x.device) * self.incx
        y_indices = torch.arange(self.n, device=x.device) * self.incy

        # Implementation note.
        out_size = y_indices.max().item() + 1
        out = torch.zeros(out_size, dtype=x.dtype, device=x.device)

        # Implementation note.
        x_flat = x.flatten()
        out[y_indices] = x_flat[x_indices]
        return out

class ModelNew(nn.Module):
    def __init__(self, n: int, incx: int, incy: int):
        super(ModelNew, self).__init__()
        self.n = n
        self.incx = incx
        self.incy = incy

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = kernel_gen_ops.ccopy(x, self.n, self.incx, self.incy)
        return out