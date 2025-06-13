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

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        """
        实现add算子功能。

        Args:
            input1: 第一个输入张量
            input2: 第二个输入张量

        Returns:
            两个输入张量的和
        """

        # 也可以使用torch的add函数
        output = torch.add(input1, input2)

        return output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input1: torch.Tensor, input2: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.add_custom(input1, input2)