from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F

import kernel_gen_ops

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, dim=-1):
        x1, x2 = x.chunk(2, dim=dim)
        return F.silu(x1) * x2

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return kernel_gen_ops.swi_glu(x, dim)