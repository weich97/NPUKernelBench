from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现Fill算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, dims: List[int], value: torch.Tensor) -> torch.Tensor:
        """
        实现Fill算子功能。

        Args:
            dims: 用于指定输出张量的形状
            value: 用于填充张量的标量值

        Returns:
            用指定标量值填充的张量
        """
        return torch.full(dims, value.item(), dtype=value.dtype)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, dims: List[int], value: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.fill(dims, value)