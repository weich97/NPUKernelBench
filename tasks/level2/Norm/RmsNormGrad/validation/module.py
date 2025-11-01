from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor, gamma: torch.Tensor) -> List[torch.Tensor]:
        normalized_dim_size = x.shape[-1] # Assuming normalization is on the last dimension

        dgamma_reduction_dims = tuple(range(dy.dim() - gamma.dim()))
        dgamma = (dy * x * rstd).sum(dim=dgamma_reduction_dims, keepdim=False)

        dx = dy * gamma * rstd - (dy * gamma * x * rstd.pow(3)).sum(dim=-1, keepdim=True) * x / normalized_dim_size

        return [dx, dgamma]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor, gamma: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.rms_norm_grad(dy, x, rstd, gamma)