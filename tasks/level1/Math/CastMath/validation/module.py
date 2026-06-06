from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()

    def forward(self, x: torch.Tensor, dst_type: int) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
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