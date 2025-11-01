from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, 
        mul0_input0: torch.Tensor,
        mul0_input1: torch.Tensor,
        mul1_input0: torch.Tensor,
        add_y: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:     
        mul_res = mul0_input0 * mul0_input1 * mul1_input0      
        reduce_mean_0 = torch.mean(mul_res, dim=1, keepdim=True)
        diff = mul_res - reduce_mean_0    
        muld_res = diff * diff
        x2 = torch.mean(muld_res, dim=1, keepdim=True)
        reduce_mean_1 = gamma / torch.sqrt(x2 + add_y)
        output = beta - reduce_mean_1 * reduce_mean_0 + reduce_mean_1 * mul_res
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, 
        mul0_input0: torch.Tensor,
        mul0_input1: torch.Tensor,
        mul1_input0: torch.Tensor,
        add_y: torch.Tensor,
        gamma: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor: 
        return kernel_gen_ops.mul_mul_reduce_mean_d_twice(mul0_input0, mul0_input1, mul1_input0, add_y, gamma, beta)