from typing import List, Optional, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self, normalized_shape: List[int]):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape

    def forward(self, dy: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
                rstd: torch.Tensor, mean: torch.Tensor, gamma: torch.Tensor,
                dsum: Optional[torch.Tensor]) -> List[torch.Tensor]:
        dy = dy.float()
        gamma = gamma.float()

        x = x1.float() + x2.float()
        d = float(torch.prod(torch.tensor(self.normalized_shape)))
        # Dimensions to reduce over (all except the last one)
        batch_axis = tuple(range(x.dim() - len(self.normalized_shape)))
        feature_axis = tuple(range(x.dim() - len(self.normalized_shape), x.dim()))


        # --- Start of translated logic ---

        pd_xl = dy * gamma
        x_hat = x - mean 

        pd_var_first_part = (-0.5) * pd_xl * x_hat * torch.pow(rstd, 3)
        pd_var = torch.sum(pd_var_first_part, dim=feature_axis, keepdim=True)

        pd_mean_first_part = torch.sum(((-1.0) * pd_xl * rstd), dim=feature_axis, keepdim=True)
        pd_mean_second_part = torch.sum(x_hat, dim=feature_axis, keepdim=True)
        pd_mean = pd_mean_first_part + pd_var * (-2.0 / d) * pd_mean_second_part

        pd_x_first_part = pd_xl * rstd
        pd_x_second_part = pd_var * (2.0 / d) * x_hat + pd_mean * (1.0 / d)

        golden_x = pd_x_first_part + pd_x_second_part
        if dsum is not None:
            golden_x += dsum.float() # Assuming input_dsum is a tensor

        golden_gamma = torch.sum(dy * x_hat * rstd, dim=batch_axis)
        golden_beta = torch.sum(dy, dim=batch_axis)

        # Cast back to original dtype
        return [
            golden_x.to(dy.dtype),
            golden_gamma,
            golden_beta
        ]


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: List[int]):
        super(ModelNew, self).__init__()
        self.normalized_shape = normalized_shape # Not directly used by the NPU kernel's forward, but for consistency

    def forward(self, dy: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor,
                rstd: torch.Tensor, mean: torch.Tensor, gamma: torch.Tensor,
                dsum: Optional[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.add_layer_norm_grad(
            dy, x1, x2, rstd, mean, gamma, dsum
        )