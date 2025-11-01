from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    简单模型，对输入张量列表中的每个张量与单个标量相乘。
    """

    def __init__(self):
        """
        初始化模型。
        标量乘法操作不需要额外参数。
        """
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
        """
        计算每个输入张量与同一个标量的乘积。

        Args:
            inputs: 输入张量列表，可以是任意形状。
            scalar: 单个标量，用于与每个输入张量相乘。

        Returns:
            与输入张量具有相同形状的输出张量列表，每个输出是输入张量与相同标量的乘积。
        """
        return [x * scalar for x in inputs]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs: List[torch.Tensor], scalar: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_mul_scalar(inputs, scalar)