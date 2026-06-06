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
        """
        Returns:
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
        """
        # Implementation note.
        dtype = x1.dtype
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        x = x1 + x2
        # Implementation note.
        variance = torch.mean(x.pow(2), dim=-1, keepdim=True)  # (..., 1)
        rstd = torch.rsqrt(variance + self.epsilon)  # 1 / sqrt(variance + eps)
        y1 = (x * rstd) * self.gamma  # Implementation note.

        # Implementation note.
        y2 = y1.to(x1.dtype)  # Implementation note.
        return [y1, y2, rstd, x]


class ModelNew(nn.Module):
    def __init__(self, gamma: torch.Tensor, epsilon: float):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.add_rms_norm_cast(x1, x2, self.gamma, self.epsilon)
