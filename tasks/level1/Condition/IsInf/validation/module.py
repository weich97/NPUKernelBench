from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Use torch.isinf, then cast the boolean result to the input tensor's dtype
        # True will become 1.0, False will become 0.0
        golden = torch.isinf(x).to(x.dtype)
        return golden


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        # Call the custom NPU operator
        return kernel_gen_ops.is_inf(x)