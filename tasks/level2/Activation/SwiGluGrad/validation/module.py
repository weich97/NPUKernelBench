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

    def forward(self, y_grad: torch.Tensor, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # 先把 x 转成 float32
        x = x.to(torch.float32)
        y_grad = y_grad.to(torch.float32)

        x1, x2 = x.chunk(2, dim=dim)

        sigmoid_x1 = torch.sigmoid(x1)                 # float32
        silu_x1 = x1 * sigmoid_x1                      # float32
        silu_prime = sigmoid_x1 * (1 + x1 * (1 - sigmoid_x1))  # float32

        grad_x1 = y_grad * x2 * silu_prime             # float32
        grad_x2 = y_grad * silu_x1                     # float32

        out = torch.cat([grad_x1, grad_x2], dim=dim)   # float32

        out = out.to(x.dtype)

        return out

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, y_grad: torch.Tensor, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return kernel_gen_ops.swi_glu_grad(y_grad, x, dim)