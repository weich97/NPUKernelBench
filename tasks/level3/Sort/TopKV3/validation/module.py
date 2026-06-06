from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

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

    def forward(self, self_tensor: torch.Tensor, k: int, dim: int, largest: bool, sorted: bool):
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        largest = bool(largest)
        sorted = bool(sorted)
        values, indices = torch.topk(self_tensor, k=k, dim=dim, largest=largest, sorted=sorted)
        return [values, indices]
    
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, self_tensor: torch.Tensor, k: int, dim: int, largest: bool, sorted: bool):
        import kernel_gen_ops

        # Implementation note.
        values, indices = kernel_gen_ops.top_kv3(self_tensor, k, dim, largest, sorted)
        return [values, indices]