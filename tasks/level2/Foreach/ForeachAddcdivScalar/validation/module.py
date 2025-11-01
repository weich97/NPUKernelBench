from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现ForeachAddcdivScalar算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
        """
        实现ForeachAddcdivScalar算子功能。

        Args:
            x1: 第一个输入张量列表
            x2: 第二个输入张量列表
            x3: 第三个输入张量列表
            scalar: 标量张量

        Returns:
            经过计算后的结果张量列表
        """
        return [x + (y / z) * scalar for x, y, z in zip(x1, x2, x3)]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_addcdiv_scalar(x1, x2, x3, scalar)