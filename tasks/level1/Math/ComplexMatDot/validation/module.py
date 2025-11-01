from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现ComplexMatDot算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, matx: torch.Tensor, maty: torch.Tensor, m: int, n: int) -> torch.Tensor:
        """
        实现ComplexMatDot算子功能。

        Args:
            matx: 第一个输入复数矩阵
            maty: 第二个输入复数矩阵
            m: 矩阵行数
            n: 矩阵列数

        Returns:
            两个输入复数矩阵逐点乘后的结果矩阵
        """
        return matx * maty


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, matx: torch.Tensor, maty: torch.Tensor, m: int, n: int) -> torch.Tensor:
        return kernel_gen_ops.complex_mat_dot(matx, maty, m, n)