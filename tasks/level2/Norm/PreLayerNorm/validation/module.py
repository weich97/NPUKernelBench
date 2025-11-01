from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, 
        x: torch.Tensor,
        y: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor:
        add = x + y
        x_shape = x.shape
        normalized_shape = (x_shape[-1],)
        output = F.layer_norm(add, normalized_shape, gamma, beta, epsilon)
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, 
        x: torch.Tensor,
        y: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor,
        epsilon: float
    ) -> torch.Tensor: 
        return kernel_gen_ops.pre_layer_norm(x, y, gamma, beta, epsilon)