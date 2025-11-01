from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现ForeachDivScalar算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
        """
        实现ForeachDivScalar算子功能。

        Args:
            x: 输入张量列表
            scalar: 标量张量

        Returns:
            输入张量列表除以标量后的结果张量列表
        """
        return [tensor / scalar for tensor in x]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_div_scalar(x, scalar)