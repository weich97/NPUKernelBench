from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn



class Model(nn.Module):
    """
    实现MSE Loss功能的模型。
    """

    def __init__(self, reduction="mean"):
        """
        初始化模型。

        Args:
            reduction: 损失计算方式，可选值为'none', 'mean', 'sum'
        """
        super(Model, self).__init__()
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        实现MSE Loss功能。

        Args:
            predict: 预测值张量
            label: 目标值张量

        Returns:
            计算得到的损失值
        """
        y = (predict - label) * (predict - label)
        if self.reduction == 'sum':
            return y.sum()
        elif self.reduction == 'mean':
            return y.mean()
        return y


class ModelNew(nn.Module):
    def __init__(self, reduction="mean"):
        """
        初始化模型。

        Args:
            reduction: 损失计算方式，可选值为'none', 'mean', 'sum'
        """
        super(ModelNew, self).__init__()
        self.reduction = reduction

    def forward(self, predict: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        使用自定义kernel实现MSE Loss功能。

        Args:
            predict: 预测值张量
            label: 目标值张量

        Returns:
            计算得到的损失值
        """
        import kernel_gen_ops
        return kernel_gen_ops.mse_loss(predict, label, self.reduction)