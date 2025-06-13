from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, dim: int) -> torch.Tensor:
        device = input1.device
        x1 = input1.cpu()
        x2 = input2.cpu()
        output = torch.cross(x1, x2, dim=dim)
        output = output.to(device)
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, dim: int) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.cross(input1, input2, dim)