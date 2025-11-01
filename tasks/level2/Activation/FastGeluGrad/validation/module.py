from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, input: torch.Tensor) -> torch.Tensor:


        attr = 1.702
        attr_opp = -attr
        attr_half = attr / 2

        abs_x = torch.abs(input)
        mul_abs_x = abs_x * attr_opp
        exp_x = torch.exp(mul_abs_x)

        add_2 = input * exp_x * attr
        exp_pn_x = torch.exp((input - abs_x) * attr)

        div_up = exp_x + add_2 + exp_pn_x
        div_down = (exp_x + 1.0) ** 2

        res = div_up / div_down
        output = dy * res

        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, dy: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.fast_gelu_grad(dy, input)