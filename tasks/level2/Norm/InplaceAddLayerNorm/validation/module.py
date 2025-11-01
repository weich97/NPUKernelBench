from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float, additional_out: bool):
        super(Model, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.epsilon = float(epsilon)
        self.additional_out = additional_out

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, bias: torch.Tensor) -> List[torch.Tensor]:
        dtype = x1.dtype
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        if bias is not None:
            bias = bias.to(torch.float32)
        x1.add_(x2)
        if bias is not None:
            x1.add_(bias)
        
        if self.additional_out:
            x2.copy_(x1)
        
        mean = x1.mean(dim=-1, keepdim=True)
        var = x1.var(dim=-1, keepdim=True, unbiased=False)
        rstd = torch.rsqrt(var + self.epsilon)

        # 计算归一化结果到x1
        x1.sub_(mean)
        x1.mul_(rstd)
        x1.mul_(self.gamma).add_(self.beta)

        x1 = x1.to(dtype)
        x2 = x2.to(dtype)
        if self.additional_out:
            return [x1, mean, rstd, x2]
        else:
            return [x1]


class ModelNew(nn.Module):
    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float, additional_out: bool):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.additional_out = additional_out

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, bias: torch.Tensor) -> List[torch.Tensor]:
        out = kernel_gen_ops.inplace_add_layer_norm(x1, x2, bias, self.gamma, self.beta, self.epsilon, self.additional_out)
        return out
