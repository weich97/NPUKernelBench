from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    实现add算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, per_token_scale: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        results = []
        start_idx = 0

        # Process each group
        for ind, end_idx in enumerate(group_list):
            m = end_idx - start_idx

            if m > 0:
                group_a = a[start_idx:end_idx]
                group_b = b[ind]

                result = torch.matmul(group_a.to(torch.int32), group_b.to(torch.int32))
                result = result.to(torch.float32) * scale[ind].unsqueeze(0).to(torch.float32) * per_token_scale[start_idx:end_idx].unsqueeze(1).to(torch.float32)
                results.append(result.flatten())

                start_idx = end_idx

        return torch.cat(results)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, per_token_scale: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.grouped_matmul_slice_m_per_token_dequant(a, b, scale, per_token_scale, group_list)