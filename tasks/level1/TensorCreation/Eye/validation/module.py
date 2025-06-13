from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.tensor, num_rows: int, num_columns: int=0, batch_shape: list[int] = [], dtype: int = 0) -> torch.Tensor:
        eye_matrix = torch.eye(num_rows, num_columns).to(input.device).to(input.dtype)
        res_shape = batch_shape + [num_rows, num_columns]

        # 广播到目标 batch 形状
        res = eye_matrix.expand(*res_shape)
        return res

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input: torch.tensor,  num_rows: int, num_columns: int=0, batch_shape: list[int] = [], dtype: int = 0) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.eye(input, num_rows, num_columns, batch_shape, dtype)