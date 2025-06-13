from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        output = torch.less_equal(input1, input2)
        output = output.to(dtype=input1.dtype)
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        # print(kernel_gen_ops.equal(input1, input2))
        result = kernel_gen_ops.less_equal(input1, input2)
        result = result.to(dtype=input1.dtype)
        return result