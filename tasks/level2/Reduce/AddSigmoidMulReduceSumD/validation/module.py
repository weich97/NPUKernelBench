from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, add_0_input0: torch.Tensor, add_0_input1: torch.Tensor, mult_0_input1: torch.Tensor, mult_1_input1: torch.Tensor, mult_2_input1: torch.Tensor) -> torch.Tensor:

        # 逐步计算
        add_res = add_0_input0 + add_0_input1
        mul1_res = add_res * mult_0_input1
        sig_res = 1 / (1 + torch.exp(-mul1_res))
        mul2_res = sig_res * mult_1_input1
        mul3_res = mul2_res * mult_2_input1
        output = torch.sum(mul3_res, dim=1)  # 按 axis=1 求和
        
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, add_0_input0: torch.Tensor, add_0_input1: torch.Tensor, mult_0_input1: torch.Tensor, mult_1_input1: torch.Tensor, mult_2_input1: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.add_sigmoid_mul_reduce_sum_d(add_0_input0, add_0_input1, mult_0_input1, mult_1_input1, mult_2_input1)