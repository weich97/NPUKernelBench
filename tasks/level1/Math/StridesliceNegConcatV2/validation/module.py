from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Implementation note.
        input = input.to(torch.float16)
        mid_col = input.shape[-1] // 2

        # Implementation note.
        output = input.clone()
        output[:, :, :, mid_col:] = -input[:, :, :, mid_col:]
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        input = input.to(torch.float16)
        return kernel_gen_ops.strideslice_neg_concat_v2(input)