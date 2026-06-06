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
        Reference implementation detail.
        """
        sparse_dim = indices.shape[0]
        if sparse_dim == 0 or indices.numel() == 0:
            return torch.tensor([], dtype=indices.dtype, device=indices.device)
        if sparse_dim == 1:
            return torch.squeeze(indices, dim=0)

        # Implementation note.
        dims = list(size)[:sparse_dim]
        
        # Implementation note.
        # Implementation note.
        strides = [1] * sparse_dim
        for i in range(sparse_dim - 2, -1, -1):
            strides[i] = strides[i+1] * dims[i+1]
        
        strides_tensor = torch.tensor(strides, dtype=indices.dtype, device=indices.device)

        # Implementation note.
        # Implementation note.
        # Implementation note.
        flatten_indices = (indices * strides_tensor.unsqueeze(1)).sum(dim=0)
        
        return flatten_indices

    def forward(self, indices: torch.Tensor, values: torch.Tensor, sparse_tensor: torch.Tensor):
        # Implementation note.
        indices_flatten = self.sparse_flatten_indices(indices, sparse_tensor.shape)

        # Implementation note.
        unique_len, unique_indices = torch.unique(indices_flatten, return_inverse=True)

        # Implementation note.
        indices = torch.transpose(indices, 0, 1)

        # Implementation note.
        return kernel_gen_ops.coalesce_sparse(unique_len, unique_indices, indices, values)