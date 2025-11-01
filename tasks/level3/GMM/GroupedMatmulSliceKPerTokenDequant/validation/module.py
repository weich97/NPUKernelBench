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
        m = a.shape[0]
        n = b.shape[1]
        offset_a = 0
        offset_b = 0
        a = a.flatten()
        b = b.flatten()
        start_idx = 0

        # Process each group
        for ind, end_idx in enumerate(group_list):
            k = end_idx - start_idx

            if k > 0:
                size_a = m * k
                size_b = k * n
                group_a_flat = a[offset_a: offset_a + size_a]
                group_b_flat = b[offset_b: offset_b + size_b]

                group_a = group_a_flat.view(k, m).transpose(0, 1)
                group_b = group_b_flat.view(k, n)
                result = torch.matmul(group_a.to(torch.int32), group_b.to(torch.int32))
                result = result.to(torch.float32) * scale[ind].unsqueeze(0).to(torch.float32) * per_token_scale[ind].unsqueeze(1).to(torch.float32)
                results.append(result.flatten())

                offset_a += size_a
                offset_b += size_b
                start_idx = end_idx
            else:
                results.append(torch.zeros([m, n], device=per_token_scale.device, dtype=per_token_scale.dtype).flatten())

        return torch.cat(results)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor, per_token_scale: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.grouped_matmul_slice_k_per_token_dequant(a, b, scale, per_token_scale, group_list)