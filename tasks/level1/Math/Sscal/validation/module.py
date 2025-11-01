from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import math

class Model(nn.Module):
    def __init__(self, n: int, incx: int):
        super(Model, self).__init__()
        self.n = n
        self.incx = incx

    def forward(self, input: torch.Tensor, alpha: float) -> torch.Tensor:
        output = input * alpha
        return output

class ModelNew(nn.Module):
    def __init__(self, n: int, incx: int):
        super(ModelNew, self).__init__()
        self.n = n
        self.incx = incx

    def forward(self, input: torch.Tensor, alpha: float) -> torch.Tensor:
        output = kernel_gen_ops.sscal(input, alpha, self.n, self.incx)
        return output