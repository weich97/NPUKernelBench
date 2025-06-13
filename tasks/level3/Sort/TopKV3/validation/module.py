from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    """
    实现TopKV3算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, self_tensor: torch.Tensor, k: int, dim: int, largest: bool, sorted: bool):
        """
        实现TopKV3算子功能。

        Args:
            self_tensor: 输入张量
            k: 计算维度上输出的极值个数
            dim: 计算维度
            largest: 布尔型，True表示计算维度上的结果应由大到小输出，False表示计算维度上的结果由小到大输出
            sorted: 布尔型，True表示输出结果排序，False表示输出结果不排序

        Returns:
            输入张量在指定维度上的k个极值及索引
        """
        largest = bool(largest)
        sorted = bool(sorted)
        values, indices = torch.topk(self_tensor, k=k, dim=dim, largest=largest, sorted=sorted)
        return [values, indices]
    
class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, self_tensor: torch.Tensor, k: int, dim: int, largest: bool, sorted: bool):
        import kernel_gen_ops

        # 调用 kernel_gen_ops.top_kv3
        values, indices = kernel_gen_ops.top_kv3(self_tensor, k, dim, largest, sorted)
        return [values, indices]