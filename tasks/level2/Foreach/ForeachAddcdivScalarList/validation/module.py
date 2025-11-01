from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现ForeachAddcdivScalarList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        实现ForeachAddcdivScalarList算子功能。

        Args:
            x1: 输入张量列表1
            x2: 输入张量列表2
            x3: 输入张量列表3
            scalars: 标量张量列表

        Returns:
            经过计算后的结果张量列表
        """
        return [t1 + (t2 / t3) * s for t1, t2, t3, s in zip(x1, x2, x3, scalars)]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalars: List[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_addcdiv_scalar_list(x1, x2, x3, scalars)