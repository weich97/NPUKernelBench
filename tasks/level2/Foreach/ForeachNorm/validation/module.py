from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
        dtype = inputs[0].dtype
        out = [torch.norm(x.float(), p=scalar.float().item()).to(dtype) for x in inputs]
        return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
        out = kernel_gen_ops.foreach_norm(tuple(inputs), scalar)
        return out
