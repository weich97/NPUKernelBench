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

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                m_list: list[int], k_list: list[int], n_list: list[int]) -> torch.Tensor:
        """
        实现add算子功能。

        Args:
            a: 第一个输入张量
            b: 第二个输入张量

        Returns:
            两个输入张量的和
        """
        results = []
        offset_a = 0
        offset_b = 0
        offset_c = 0
        # Process each group
        for ind, (m, k, n) in enumerate(zip(m_list, k_list, n_list)):

            size_a = m * k
            size_b = k * n
            size_c = m * n
            group_a_flat = a[offset_a: offset_a + size_a]
            group_b_flat = b[offset_b: offset_b + size_b]
            group_c_flat = c[offset_c: offset_c + size_c]

            group_a = group_a_flat.view(m, k)
            group_b = group_b_flat.view(k, n)
            group_c = group_c_flat.view(m, n)
            result = (torch.matmul(alpha[ind] * group_a.to(torch.float32), group_b.to(torch.float32))
                      + beta[ind] * group_c.to(torch.float32)).flatten()
            results.append(result)

            offset_a += size_a
            offset_b += size_b
            offset_c += size_c

        return torch.cat(results)


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                m_list: list[int], k_list: list[int], n_list: list[int]) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.group_gemm(a, b, c, alpha, beta, m_list, k_list, n_list)