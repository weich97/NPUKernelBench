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

    def forward(self, x: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        return [t - s for t, s in zip(x, scalars)]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_sub_scalar_list(x, scalars)