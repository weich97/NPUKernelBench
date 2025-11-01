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
        # PyTorch equivalent: torch.argmin(torch.abs(x)) + 1 for 1-indexed output
        
        # Extract the elements according to n and incx
        if self.incx == 1:
            # If incx is 1, consider the first `n` elements
            sliced_x = x[:self.n]
        else:
            # Manually extract elements with stride `incx`
            end_val = float(self.n) * float(self.incx)
            step_val = float(self.incx)
            
            indices = torch.arange(0.0, end_val, step_val, device=x.device)
            indices = indices.long()
            sliced_x = x.index_select(0, indices)
            
        # Compute absolute values
        abs_x = torch.abs(sliced_x)
        
        # Find the index of the minimum absolute value
        # torch.argmin returns 0-indexed.
        argmin_idx = torch.argmin(abs_x)
        
        # According to typical BLAS ISAMIN output, add 1 for 1-indexed output
        golden = argmin_idx + 1
        
        # Ensure the output is a scalar tensor with int32 dtype
        return golden.to(torch.int32)


class ModelNew(nn.Module):
    def __init__(self, n: int, incx: int):
        super(ModelNew, self).__init__()
        self.n = n
        self.incx = incx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom NPU operator
        return kernel_gen_ops.isamin(x, self.n, self.incx)