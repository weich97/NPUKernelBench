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
        x_add = x1 + x2 # First, perform the addition
        
        # Then, perform RMSNorm on the result of the addition
        # The RmsNorm formula is: y = x_add * rstd * gamma
        # Where rstd = 1 / sqrt(mean(x_add^2) + epsilon)
        
        # Calculate rstd (Reciprocal Standard Deviation) for the added tensor
        # Assuming RMSNorm is applied over the last dimension, consistent with typical LayerNorm/RMSNorm behavior
        rstd = torch.rsqrt(x_add.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        
        # Apply gamma
        y_out = x_add * rstd * self.gamma
        
        return [y_out, rstd, x_add]

class ModelNew(nn.Module):
    def __init__(self, gamma: torch.Tensor, epsilon: float):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.add_rms_norm(x1, x2, self.gamma, self.epsilon)