from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


import torch
import torch.nn as nn

import torch
import torch.nn as nn

import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Reference implementation detail.
    Reference implementation detail.
    1. tmp = sigmoid(x1 * t1)
    2. sel = where(tmp < t2, tmp, 2*tmp)
    3. res = sel * x2 * t3
    """

    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, 
                t1: float, t2: float, 
                t3: float) -> torch.Tensor:
        # Implementation note.
        tmp = torch.sigmoid(x1 * t1)
        
        # Implementation note.
        sel = torch.where(tmp < t2, tmp, 2 * tmp)
        
        # Implementation note.
        x2_flat_dim = torch.prod(torch.tensor(x2.shape[1:])).item()
        # Implementation note.
        x2_reshaped = x2.reshape(1, x2_flat_dim)
        
        # Implementation note.
        sel_reshaped = sel.reshape(-1, x2_flat_dim)
        
        # Implementation note.
        res = sel_reshaped * x2_reshaped * t3
        
        # Implementation note.
        output_shape = (x1.shape[0],) + x2.shape[1:]
        res = res.reshape(output_shape)
        
        return res


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, t1: float, t2: float, t3: float) -> torch.Tensor:
        
        output = kernel_gen_ops.mul_sigmoid(x1, x2, t1, t2, t3)
        
        return output