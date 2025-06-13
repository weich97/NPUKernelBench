from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    实现GELU梯度算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        实现GELU梯度算子功能。

        Args:
            dy: 梯度输入张量
            x: 原始输入张量

        Returns:
            GELU梯度计算结果
        """
        # 使用数学公式直接计算GELU梯度
        c0 = -0.0713548162726002527220
        c1 = -1.595769121605730711759
        c2 = 0.2140644488178007
        c3 = 1.595769121605730711759

        x_square = x * x
        px_arg = (x_square * c0 + c1) * x
        px = torch.exp(px_arg)
        res0 = (x_square * c2 + c3) * x
        t_denominator = px + 1.0
        t = 1.0 / t_denominator
        resp_intermediate = px * res0 * t.pow(2) + t
        resp = torch.nan_to_num(resp_intermediate, nan=0.0)
        z = dy * resp

        return z


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        使用自定义kernel实现GELU梯度算子功能。

        Args:
            dy: 梯度输入张量
            x: 原始输入张量

        Returns:
            GELU梯度计算结果
        """
        import kernel_gen_ops
        # 计算GELU前向结果
        y = torch.nn.functional.gelu(x)

        # 使用自定义算子计算GELU梯度
        return kernel_gen_ops.gelu_grad(dy, x, y)