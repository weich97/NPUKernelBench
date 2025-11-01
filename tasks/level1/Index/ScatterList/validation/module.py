from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import math

class Model(nn.Module):
    """
    实现ScatterList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, varRef: list[torch.Tensor], indice: torch.Tensor, updates: torch.Tensor, mask: Optional[torch.Tensor], reduce: str, axis: int) -> torch.Tensor:
        """
        实现ScatterList算子功能。

        Args:
            varRef: 第一个输入张量
            indice: 索引张量
            updates: 更新张量
            mask: 可选的掩码张量
            reduce: 规约操作类型
            axis: 指定的轴

        Returns:
            经过ScatterList操作后的结果张量
        """
        for i in range(len(varRef)):
            if mask[i] == False: continue
            dest_block_slice = slice(indice[i][0], indice[i][0] + indice[i][1])
            source_block_slice = slice(0, indice[i][1])
            
            num_dims = varRef[i].ndim
            dest_slicer = [slice(None)] * num_dims
            src_slicer = [slice(None)] * num_dims

            dest_slicer[axis] = dest_block_slice
            src_slicer[axis] = source_block_slice
            
            dest_slicer = tuple(dest_slicer)
            src_slicer = tuple(src_slicer)
            
            source_block = updates[i][src_slicer]
            varRef[i][dest_slicer] = source_block

        return varRef

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, varRef: list[torch.Tensor], indice: torch.Tensor, updates: torch.Tensor, mask: Optional[torch.Tensor], reduce: str, axis: int) -> torch.Tensor:
        return kernel_gen_ops.scatter_list(varRef, indice, updates, mask, reduce, axis)