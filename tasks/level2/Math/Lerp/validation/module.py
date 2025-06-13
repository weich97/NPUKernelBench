from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, start: torch.Tensor, end: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        output = torch.lerp(start, end, weight)
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, start: torch.Tensor, end: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.lerp(start, end, weight)