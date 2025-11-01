from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现Cast算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dst_type: int) -> torch.Tensor:
        """
        实现Cast算子功能。

        Args:
            x: 输入张量
            dst_type: 目标数据类型的标识

        Returns:
            转换为目标数据类型后的张量
        """
        return x.to(self._get_dtype(dst_type))

    def _get_dtype(self, dst_type):
        type_map = {
            0: torch.float32,
            1: torch.float16,
            2: torch.int8,
            3: torch.int32,
            4: torch.uint8,
            6: torch.int16,
            9: torch.int64,
            12: torch.bool,
            27: torch.bfloat16
        }
        return type_map.get(dst_type, torch.float32)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, dst_type: int) -> torch.Tensor:
        return kernel_gen_ops.cast_math(x, dst_type)