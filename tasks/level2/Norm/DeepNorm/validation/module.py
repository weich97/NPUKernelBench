from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self, beta: torch.Tensor, gamma: torch.Tensor, alpha: float, epsilon: float):
        super(Model, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, gx: torch.Tensor) -> List[torch.Tensor]:
        # Convert inputs to float32 for intermediate calculations
        x_fp32 = x.to(torch.float32)
        gx_fp32 = gx.to(torch.float32)
        beta_fp32 = self.beta.to(torch.float32)
        gamma_fp32 = self.gamma.to(torch.float32)

        # Compute x_add in float32
        x_add = x_fp32 * self.alpha + gx_fp32

        # Compute mean and variance in float32
        mean = x_add.mean(-1, keepdim=True)
        diff = x_add - mean
        variance = diff.pow(2).mean(-1, keepdim=True)
        rstd = torch.rsqrt(variance + self.epsilon)

        # Compute y_out in float32
        y_out = gamma_fp32 * diff * rstd + beta_fp32

        # Convert outputs back to original dtype
        mean = mean.to(x.dtype)
        rstd = rstd.to(x.dtype)
        y_out = y_out.to(x.dtype)

        return [mean, rstd, y_out]


class ModelNew(nn.Module):
    def __init__(self, beta: torch.Tensor, gamma: torch.Tensor, alpha: float, epsilon: float):
        super(ModelNew, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

    def forward(self, x: torch.Tensor, gx: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.deep_norm(x, gx, self.beta, self.gamma, self.alpha, self.epsilon)