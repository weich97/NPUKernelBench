from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()

    def forward(self, inputs1: List[torch.Tensor], inputs2: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        return [torch.minimum(x, y) for x, y in zip(inputs1, inputs2)]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs1: List[torch.Tensor], inputs2: List[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_minimum_list(inputs1, inputs2)