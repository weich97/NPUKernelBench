from typing import List, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self, n: int, incx: int):
        super(Model, self).__init__()
        self.n = n     # Number of elements to consider
        self.incx = incx # Stride for x (increment for x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # For Icamin: Find the index of the MINIMUM absolute sum of real and imaginary parts.
        # PyTorch equivalent: torch.argmin(torch.abs(x.real) + torch.abs(x.imag)) + 1
        
        # Extract the elements according to n and incx
        if self.incx == 1:
            sliced_x = x[:self.n]
            # original_indices_selected for incx=1 is simply arange(0, self.n)
            original_indices_selected = torch.arange(0, self.n, device=x.device).long()
        else:
            end_val = float(self.n) * float(self.incx)
            step_val = float(self.incx)
            
            indices = torch.arange(0.0, end_val, step_val, device=x.device).long()
            sliced_x = x.index_select(0, indices)

            # Store the actual original indices to map back later
            original_indices_selected = indices # These are the original 0-based indices that were selected
            
        # Compute the sum of absolute values of real and imaginary parts
        abs_real_imag = torch.abs(sliced_x.real) + torch.abs(sliced_x.imag)
        
        # Find the index of the MINIMUM sum of absolute values within the sliced_x
        # torch.argmin returns the index of the FIRST occurrence if there are ties.
        argmin_idx_in_sliced = torch.argmin(abs_real_imag)
        
        # Now, map this index back to the original tensor's 0-based index
        golden_0_based_idx = original_indices_selected[argmin_idx_in_sliced]
        
        # Add 1 for 1-based indexing as per BLAS ICAMIN standard
        golden = golden_0_based_idx + 1
        
        # Ensure the output is a scalar tensor with int32 dtype
        return golden.to(torch.int32)


class ModelNew(nn.Module):
    def __init__(self, n: int, incx: int):
        super(ModelNew, self).__init__()
        self.n = n
        self.incx = incx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom NPU operator
        return kernel_gen_ops.icamin(x, self.n, self.incx)