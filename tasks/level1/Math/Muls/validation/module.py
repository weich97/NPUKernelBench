from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现Muls算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        实现Muls算子功能。

        Args:
            x: 输入张量
            value: 标量张量

        Returns:
            输入张量与标量张量相乘后的结果张量（float16）
        """

        # 计算
        out = x * value

        # 结果已经是 float16
        return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.muls(x, value)