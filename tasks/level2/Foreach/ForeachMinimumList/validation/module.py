from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现ForeachminimumList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, inputs1: List[torch.Tensor], inputs2: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        实现ForeachminimumList算子功能。

        Args:
            input1: 第一个输入张量列表
            input2: 第二个输入张量列表

        Returns:
            两个输入张量列表逐元素比较后的最大值张量列表
        """
        return [torch.minimum(x, y) for x, y in zip(inputs1, inputs2)]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs1: List[torch.Tensor], inputs2: List[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_minimum_list(inputs1, inputs2)