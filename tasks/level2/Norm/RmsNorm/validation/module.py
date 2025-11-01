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

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Reference implementation for RmsNorm
        # Based on the provided golden data generation example
        out = x * torch.rsqrt(x.type(torch.float32).pow(2).mean(-1, keepdim=True) + self.epsilon).type(torch.float32)
        golden = out * self.gamma.type(torch.float32)
        return [golden]

class ModelNew(nn.Module):
    def __init__(self, gamma: torch.Tensor, epsilon: float):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.rms_norm(x, self.gamma, self.epsilon)