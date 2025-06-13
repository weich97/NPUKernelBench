from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn



class Model(nn.Module):
    """
    简单模型，对两个输入张量列表中对应位置的张量进行相乘操作。
    """

    def __init__(self):
        """
        初始化模型。
        张量乘法操作不需要额外参数。
        """
        super(Model, self).__init__()

    def forward(self, inputs1: List[torch.Tensor], inputs2: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        计算两个张量列表中对应位置张量的乘积。

        Args:
            inputs1: 第一个输入张量列表，可以是任意形状。
            inputs2: 第二个输入张量列表，可以是任意形状。

        Returns:
            与输入张量具有相同形状的输出张量列表，每个输出是对应位置两个输入张量的乘积。
        """
        return [torch.mul(x, y) for x, y in zip(inputs1, inputs2)]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs1: List[torch.Tensor], inputs2: List[torch.Tensor]) -> List[torch.Tensor]:
        import kernel_gen_ops
        return kernel_gen_ops.foreach_mul_list(inputs1, inputs2)