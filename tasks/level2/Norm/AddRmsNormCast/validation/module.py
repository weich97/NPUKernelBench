from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops



class Model(nn.Module):

    def __init__(self, gamma: torch.Tensor, epsilon: float):
        super(Model, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            y1: FP32 归一化结果
            y2: FP16/BF16 归一化结果
            rstd: 倒数标准差 (FP32)
            x: 残差加法后的中间结果 (FP16/BF16)
        """
        # 1. 残差加法 (x = x1 + x2)
        dtype = x1.dtype
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        x = x1 + x2
        # 2. 计算 RMSNorm
        variance = torch.mean(x.pow(2), dim=-1, keepdim=True)  # (..., 1)
        rstd = torch.rsqrt(variance + self.epsilon)  # 1 / sqrt(variance + eps)
        y1 = (x * rstd) * self.gamma  # FP32 结果

        # 3. 数据类型转换
        y2 = y1.to(x1.dtype)     # 转回 FP16/BF16        
        return [y1, y2, rstd, x]


class ModelNew(nn.Module):
    def __init__(self, gamma: torch.Tensor, epsilon: float):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.epsilon = epsilon

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.add_rms_norm_cast(x1, x2, self.gamma, self.epsilon)
