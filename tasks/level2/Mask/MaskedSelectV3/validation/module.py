from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import math

class Model(nn.Module):
    """
    实现MaskSelectV3算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        实现MaskSelectV3算子功能。

        Args:
            input_tensor: 输入张量
            mask: 布尔掩码张量

        Returns:
            根据掩码选择元素后的一维张量
        """

        output = torch.masked_select(input_tensor, mask)

        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.masked_select_v3(input_tensor, mask)