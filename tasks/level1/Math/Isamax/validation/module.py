from typing import List, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self, n: int, incx: int):
        super(Model, self).__init__()
        self.n = n
        self.incx = incx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Extract the elements according to n and incx
        if self.incx == 1:
            # If incx is 1, consider the first `n` elements
            sliced_x = x[:self.n]
        else:
            # Manually extract elements with stride `incx`
            # --- FIX START ---
            # Explicitly cast n and incx to int or float to prevent string conversion issues
            # Using float() ensures compatibility even if intermediate values become very large.
            end_val = float(self.n) * float(self.incx)
            step_val = float(self.incx)
            
            indices = torch.arange(0.0, end_val, step_val, device=x.device)
            # Ensure indices are long for index_select
            indices = indices.long()
            # --- FIX END ---
            sliced_x = x.index_select(0, indices)
            
        # Compute absolute values
        abs_x = torch.abs(sliced_x)
        
        # Find the index of the maximum absolute value
        argmax_idx = torch.argmax(abs_x)
        
        # According to golden data, add 1 (1-indexed output often in BLAS)
        golden = argmax_idx + 1
        
        # Ensure the output is a scalar tensor with int32 dtype as per golden.bin
        return golden.to(torch.int32)


class ModelNew(nn.Module):
    def __init__(self, n: int, incx: int):
        super(ModelNew, self).__init__()
        self.n = n
        self.incx = incx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.isamax(x, self.n, self.incx)