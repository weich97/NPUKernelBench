from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import torch.nn.functional as F

from typing import Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        x1, x2 = x.chunk(2, dim=-1)
        gelu_out = F.gelu(x1)
        out = gelu_out * x2
        return [out, gelu_out]

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor):
        return kernel_gen_ops.ge_glu_v2(x)