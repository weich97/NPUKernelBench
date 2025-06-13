from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.swi_glu(x)