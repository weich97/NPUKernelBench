from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, mulscalar1: torch.Tensor, mulscalar2: torch.Tensor, mulscalar3: torch.Tensor) -> torch.Tensor:
        mul1_res = input * mulscalar1
        sigmoid_res = 1 / (1 + torch.exp(-mul1_res))
        mul_2_res = sigmoid_res * mulscalar2
        output = mul_2_res + mulscalar3
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor, mulscalar1: torch.Tensor, mulscalar2: torch.Tensor, mulscalar3: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.mul_sigmoid_mul_add_custom(input, mulscalar1, mulscalar2, mulscalar3)