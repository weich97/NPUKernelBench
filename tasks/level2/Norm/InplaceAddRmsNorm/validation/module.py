from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops



class Model(nn.Module):

    def __init__(self, gamma: torch.Tensor, epsilon: float):
        super(Model, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        dtype = x1.dtype
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)
        # 1. 复用x1的内存存储加法结果 (y = x1 + x2)
        x1.add_(x2)  # y_out 复用x1的内存

        # 2. 复用x2的内存存储中间变量 (x = x2)
        x2.copy_(x1)  # 现在x2存储加法结果

        # 3. 计算RMSNorm
        rstd = torch.rsqrt(x2.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)  # x2是加法结果
        x1.mul_(rstd).mul_(self.gamma)  # y_out最终结果存在x1

        #    y_out已在x1的内存中，x_add在x2的内存中
        x1 = x1.to(dtype)
        x2 = x2.to(dtype)
        return [x1, rstd, x2]


class ModelNew(nn.Module):
    def __init__(self, gamma: torch.Tensor, epsilon: float):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.inplace_add_rms_norm(x1, x2, self.gamma, self.epsilon)
