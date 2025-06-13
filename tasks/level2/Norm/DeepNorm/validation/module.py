from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):

    def __init__(self, beta: torch.Tensor, gamma: torch.Tensor, alpha: float, epsilon: float):
        super(Model, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, gx: torch.Tensor) -> List[torch.Tensor]:
        # Based on gen_golden_data_simple logic
        x_add = x * self.alpha + gx
        mean = x_add.mean(-1, keepdim=True)
        diff = x_add - mean
        variance = diff.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)
        y_out = self.gamma * diff * rstd + self.beta

        return [mean, rstd, y_out] # Outputs as specified by aclnnDeepNormGetWorkspaceSize: meanOut, rstdOut, yOut


class ModelNew(nn.Module):
    def __init__(self, beta: torch.Tensor, gamma: torch.Tensor, alpha: float, epsilon: float):
        super(ModelNew, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, gx: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops
        return kernel_gen_ops.deep_norm(x, gx, self.beta, self.gamma, self.alpha, self.epsilon)