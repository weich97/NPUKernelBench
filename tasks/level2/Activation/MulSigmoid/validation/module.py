from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    实现MulSigmoid算子功能的动态形状处理模型
    计算公式：
    1. tmp = sigmoid(x1 * t1)
    2. sel = where(tmp < t2, tmp, 2*tmp)
    3. res = sel * x2 * t3
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                t1: float, t2: float, 
                t3: float) -> torch.Tensor:
        # 步骤1: 计算 tmp = sigmoid(x1 * t1)
        tmp = torch.sigmoid(x1 * t1)
        
        # 步骤2: 根据条件选择 tmp 或 2*tmp 作为 sel
        sel = torch.where(tmp < t2, tmp, 2 * tmp)
        
        # 计算x2的展平维度（保持第一维，其余维度展平）
        x2_flat_dim = torch.prod(torch.tensor(x2.shape[1:])).item()
        # 展平x2的后几维，形状变为[1, x2_flat_dim]
        x2_reshaped = x2.reshape(1, x2_flat_dim)
        
        # 展平sel的后几维，使其与x2展平后的维度匹配
        sel_reshaped = sel.reshape(-1, x2_flat_dim)
        
        # 步骤3: 计算sel * x2 * t3
        res = sel_reshaped * x2_reshaped * t3
        
        # 恢复输出形状：(x1的第一维, x2的第二维, x2的第三维, ...)
        output_shape = (x1.shape[0],) + x2.shape[1:]
        res = res.reshape(output_shape)
        
        return res


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, t1: float, t2: float, t3: float) -> torch.Tensor:
        
        output = kernel_gen_ops.mul_sigmoid(x1, x2, t1, t2, t3)
        
        return output