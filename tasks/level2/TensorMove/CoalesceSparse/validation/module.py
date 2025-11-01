from typing import Tuple, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
from typing import List
import kernel_gen_ops
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, indices: torch.Tensor, values: torch.Tensor, sparse_tensor: torch.Tensor):
        sparse_tensor = sparse_tensor.coalesce()
        new_indices = sparse_tensor.indices()
        new_values = sparse_tensor.values()
        new_indices = torch.transpose(new_indices, 0, 1)
        
        return [new_indices, new_values]

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def sparse_flatten_indices(self, indices, size):
        """
        使用正确且高效的张量操作将多维稀疏索引展平为一维。
        """
        sparse_dim = indices.shape[0]
        if sparse_dim == 0 or indices.numel() == 0:
            return torch.tensor([], dtype=indices.dtype, device=indices.device)
        if sparse_dim == 1:
            return torch.squeeze(indices, dim=0)

        # 1. 获取稀疏维度的大小
        dims = list(size)[:sparse_dim]
        
        # 2. 正确计算每个维度的步长 (strides)
        # 例如, 对于 shape (d1, d2, d3), 步长应为 (d2*d3, d3, 1)
        strides = [1] * sparse_dim
        for i in range(sparse_dim - 2, -1, -1):
            strides[i] = strides[i+1] * dims[i+1]
        
        strides_tensor = torch.tensor(strides, dtype=indices.dtype, device=indices.device)

        # 3. 将多维索引乘以步长并求和，得到一维索引
        # indices 的形状是 [dims, nnz], strides_tensor 的形状是 [dims]
        # 我们需要将 strides_tensor 调整为 [dims, 1] 以便进行广播乘法
        flatten_indices = (indices * strides_tensor.unsqueeze(1)).sum(dim=0)
        
        return flatten_indices

    def forward(self, indices: torch.Tensor, values: torch.Tensor, sparse_tensor: torch.Tensor):
        # 1. 将多维的`indices`展平为一维
        indices_flatten = self.sparse_flatten_indices(indices, sparse_tensor.shape)

        # 2. 获取唯一的一维索引值（unique_len）和逆映射（unique_indices）
        unique_len, unique_indices = torch.unique(indices_flatten, return_inverse=True)

        # 3. 将`indices`的形状从 [dims, nnz] 转置为 [nnz, dims]，以符合自定义算子的输入要求
        indices = torch.transpose(indices, 0, 1)

        # 4. 调用自定义的`coalesce_sparse`算子
        return kernel_gen_ops.coalesce_sparse(unique_len, unique_indices, indices, values)