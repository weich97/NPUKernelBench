from typing import List, Optional, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, eps: float, activate_swish: bool, swish_scale: float):
        super(Model, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.activate_swish = activate_swish
        self.swish_scale = swish_scale

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> List[torch.Tensor]:
        input_dtype = x.dtype
        N = x.shape[0]
        C = self.num_channels
        
        remaining_dims_product = 1
        for size in x.shape[2:]:
            remaining_dims_product *= size
        HxW = remaining_dims_product
        
        x_fp32 = x.to(torch.float32)
        gamma_fp32 = gamma.to(torch.float32)
        beta_fp32 = beta.to(torch.float32)
        
        gn_out, mean_out, rstd_out = torch.ops.aten.native_group_norm(
            x_fp32,
            gamma_fp32,
            beta_fp32,
            N,
            C,
            HxW,
            self.num_groups,
            self.eps
        )

        if self.activate_swish:
            sigmoid_arg = self.swish_scale * gn_out
            swish_out = gn_out * torch.sigmoid(sigmoid_arg)
            final_out = swish_out
        else:
            final_out = gn_out

        return [final_out.to(input_dtype)]


class ModelNew(nn.Module):
    def __init__(self, num_channels: int, num_groups: int, eps: float, activate_swish: bool, swish_scale: float):
        super(ModelNew, self).__init__()
        self.num_channels = num_channels
        self.num_groups = num_groups
        self.eps = eps
        self.activate_swish = activate_swish
        self.swish_scale = swish_scale

    def forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> List[torch.Tensor]:
        # Call the NPU kernel and extract only the yOut for comparison
        y_out_npu, _, _ = kernel_gen_ops.group_norm_swish(
            x, gamma, beta, self.num_groups,
            "",
            self.eps,
            self.activate_swish,
            self.swish_scale
        )
        return [y_out_npu]