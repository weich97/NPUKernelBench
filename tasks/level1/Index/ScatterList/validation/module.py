from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops
import math

class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()

    def forward(self, varRef: list[torch.Tensor], indice: torch.Tensor, updates: torch.Tensor, mask: Optional[torch.Tensor], reduce: str, axis: int) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
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