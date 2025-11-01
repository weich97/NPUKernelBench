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
        # PyTorch equivalent: torch.sqrt(torch.sum(x**2)) or torch.linalg.norm(x)
        
        # Extract the elements according to n and incx
        if self.incx == 1:
            sliced_x = x[:self.n]
        else:
            end_val = float(self.n) * float(self.incx)
            step_val = float(self.incx)
            
            indices = torch.arange(0.0, end_val, step_val, device=x.device).long()
            sliced_x = x.index_select(0, indices)
            
        # Compute the sum of squares
        sum_of_squares = torch.sum(sliced_x**2)
        
        # Compute the square root
        golden = torch.sqrt(sum_of_squares)
        
        # Ensure the output is a scalar tensor with the same dtype as x
        return golden.to(x.dtype)


class ModelNew(nn.Module):
    def __init__(self, n: int, incx: int):
        super(ModelNew, self).__init__()
        self.n = n
        self.incx = incx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Call the custom NPU operator
        return kernel_gen_ops.snrm2(x, self.n, self.incx)