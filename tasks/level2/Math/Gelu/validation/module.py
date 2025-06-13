from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    实现GELU算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        实现GELU算子功能。

        Args:
            input_tensor: 输入张量

        Returns:
            应用GELU激活函数后的张量
        """
        # 标准GELU实现: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        return torch.nn.functional.gelu(input_tensor)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        使用自定义kernel实现GELU算子功能。

        Args:
            input_tensor: 输入张量

        Returns:
            应用GELU激活函数后的张量
        """
        import kernel_gen_ops
        return kernel_gen_ops.gelu(input_tensor)