from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现ForeachDivScalarList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        实现ForeachDivScalarList算子功能。

        Args:
            x: 输入张量列表
            scalars: 标量张量列表

        Returns:
            输入张量列表与标量张量列表逐元素相除后的结果张量列表
        """
        return [tensor / scalar for tensor, scalar in zip(x, scalars)]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_div_scalar_list(x, scalars)