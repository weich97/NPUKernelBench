from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    实现add算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: torch.float32, beta: torch.float32) -> torch.Tensor:
        """
        实现add算子功能。

        Args:
            a: 第一个输入张量
            b: 第二个输入张量

        Returns:
            两个输入张量的和
        """
        output = alpha * (a @ x) + beta * y
        return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, a: torch.Tensor, x: torch.Tensor, y: torch.Tensor, alpha: torch.float32, beta: torch.float32) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.gemv(a, x, y, alpha, beta)