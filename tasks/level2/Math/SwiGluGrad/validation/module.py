from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, y_grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        sigmoid_x1 = torch.sigmoid(x1)
        silu_x1 = x1 * sigmoid_x1
        silu_prime = sigmoid_x1 * (1 + x1 * (1 - sigmoid_x1))
        grad_x1 = y_grad * x2 * silu_prime
        grad_x2 = y_grad * silu_x1
        return torch.cat([grad_x1, grad_x2], dim=-1)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, y_grad: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.swi_glu_grad(y_grad, x)