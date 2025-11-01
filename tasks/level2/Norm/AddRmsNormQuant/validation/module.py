from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn



class Model(nn.Module):

    def __init__(self, 
    gamma: torch.Tensor, 
    scales1: torch.Tensor, 
    scales2: Optional[torch.Tensor] = None, 
    zero_points1: Optional[torch.Tensor] = None,
    zero_points2: Optional[torch.Tensor] = None,   
    axis: int = -1,
    epsilon: float = 1e-6,
    div_mode: bool = True):
    
        super(Model, self).__init__()
        self.gamma = gamma.to(torch.float32).to('cpu')
        self.scales1 = scales1.to(torch.float32).to('cpu')
        self.scales2 = scales2

        self.zero_points1 = zero_points1.to(torch.float32).to('cpu')
        self.zero_points2 = zero_points2

        self.axis = axis
        self.epsilon = epsilon
        self.div_mode = div_mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        x1 = x1.to(torch.float32)
        x2 = x2.to(torch.float32)

        x = x1 + x2
        
        rms = torch.sqrt(x.pow(2).mean(dim=self.axis, keepdim=True) + self.epsilon)

        if self.div_mode:
            x_norm = x / rms
        else:
            x_norm = x * torch.rsqrt(rms + self.epsilon)
        
        y = x_norm * self.gamma

        if not self.div_mode:
            self.scales1 = 1.0 / self.scales1

        y1 = torch.quantize_per_channel(y, self.scales1, self.zero_points1, len(x1.shape) - len(self.gamma.shape), torch.qint8)
            
        return [y1.int_repr()]


class ModelNew(nn.Module):
    def __init__(self, 
    gamma: torch.Tensor, 
    scales1: torch.Tensor, 
    scales2: Optional[torch.Tensor] = None, 
    zero_points1: Optional[torch.Tensor] = None,
    zero_points2: Optional[torch.Tensor] = None,   
    axis: int = -1,
    epsilon: float = 1e-6,
    div_mode: bool = True):
    
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.scales1 = scales1
        self.scales2 = scales2

        self.zero_points1 = zero_points1
        self.zero_points2 = zero_points2

        self.axis = axis
        self.epsilon = epsilon
        self.div_mode = div_mode

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops
        return kernel_gen_ops.add_rms_norm_quant(x1, x2, self.gamma, self.scales1, self.scales2, self.zero_points1, self.zero_points2,self.axis, self.epsilon, self.div_mode)
