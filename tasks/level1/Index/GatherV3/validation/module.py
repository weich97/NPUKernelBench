from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import math

class Model(nn.Module):
    """
    实现GatherV3算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, self_tensor: torch.Tensor, axis: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """
        实现GatherV3算子功能。

        Args:
            self_tensor: 第一个输入张量
            indices: 索引张量
            axis: 轴张量

        Returns:
            GatherV3算子的输出
        """

        output = torch.index_select(self_tensor, dim=axis, index=indices)

        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, self_tensor: torch.Tensor, axis: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.gather_v3(self_tensor, axis, indices)