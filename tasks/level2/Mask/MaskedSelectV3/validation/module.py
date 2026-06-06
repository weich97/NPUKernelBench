from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import math

class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()

    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """

        output = torch.masked_select(input_tensor, mask)

        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.masked_select_v3(input_tensor, mask)