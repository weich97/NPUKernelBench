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

    def forward(self, a: torch.Tensor, b: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        results = []
        k = a.shape[1]
        offset_a = 0
        a = a.flatten()
        start_idx = 0

        # Process each group
        for ind, end_idx in enumerate(group_list):
            m = end_idx - start_idx

            if m > 0:
                size_a = m * k
                group_a_flat = a[offset_a: offset_a + size_a]
                group_b = b[ind]

                group_a = group_a_flat.view(m, k)
                result = torch.matmul(group_a, group_b).flatten()
                results.append(result)

                offset_a += size_a
                start_idx = end_idx

        return torch.cat(results)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.grouped_matmul_slice_m(a, b, group_list)