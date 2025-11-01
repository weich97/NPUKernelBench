from typing import List, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self, n: int, incx: int):
        super(Model, self).__init__()
        self.n = n     # Number of elements to consider
        self.incx = incx # Stride for x (increment for x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sliced = x[:self.n] if self.incx == 1 else \
                 x.index_select(0, torch.arange(0, self.n * self.incx, self.incx, device=x.device))
        
        out = torch.sum(torch.abs(sliced))
        
        return out


class ModelNew(nn.Module):
    def __init__(self, n: int, incx: int):
        super(ModelNew, self).__init__()
        self.n = n
        self.incx = incx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom NPU operator
        return kernel_gen_ops.sasum(x, self.n, self.incx)