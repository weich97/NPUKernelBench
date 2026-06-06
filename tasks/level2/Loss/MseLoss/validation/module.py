from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn



class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self, reduction="mean"):
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
        """
        super(Model, self).__init__()
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        y = (predict - label) * (predict - label)
        if self.reduction == 'sum':
            return y.sum()
        elif self.reduction == 'mean':
            return y.mean()
        return y


class ModelNew(nn.Module):
    def __init__(self, reduction="mean"):
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
        """
        super(ModelNew, self).__init__()
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        import kernel_gen_ops
        return kernel_gen_ops.mse_loss(predict, label, self.reduction)