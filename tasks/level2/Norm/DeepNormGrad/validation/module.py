from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, gx: torch.Tensor,
                gamma: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, alpha: float) -> List[torch.Tensor]:
        # Convert all inputs to float32 for intermediate calculations
        dy_fp32 = dy.to(torch.float32)
        x_fp32 = x.to(torch.float32)
        gx_fp32 = gx.to(torch.float32)
        gamma_fp32 = gamma.to(torch.float32)
        mean_fp32 = mean.to(torch.float32)
        rstd_fp32 = rstd.to(torch.float32)

        # Determine the normalization dimension (D) for the mean/variance calculations
        D = float(torch.prod(torch.tensor(gamma_fp32.shape)))  # Or gamma.shape[-1] if gamma is 1D

        # Calculate intermediate terms
        tmpone = dy_fp32 * gamma_fp32
        tmptwo = alpha * x_fp32 + gx_fp32 - mean_fp32

        # Calculate d_var and d_mean (sum over the normalized dimension)
        reduction_dims = tuple(range(x_fp32.dim() - gamma_fp32.dim(), x_fp32.dim()))
        d_var = torch.sum(-0.5 * tmpone * tmptwo * rstd_fp32.pow(3), dim=reduction_dims, keepdim=True)
        d_mean = torch.sum(-1.0 * tmpone * rstd_fp32, dim=reduction_dims, keepdim=True)

        # Calculate d_gx
        dgx = tmpone * rstd_fp32 + (2.0 / D) * d_var * tmptwo + (1.0 / D) * d_mean

        # Calculate d_x
        dx = alpha * dgx

        # Calculate d_beta (sum over all dimensions except normalized_shape)
        d_reduction_dims_for_gamma_beta = tuple(range(dy_fp32.dim() - gamma_fp32.dim()))
        dbeta = torch.sum(dy_fp32, dim=d_reduction_dims_for_gamma_beta, keepdim=False)

        # Calculate d_gamma (sum over all dimensions except normalized_shape)
        dgamma = torch.sum(dy_fp32 * rstd_fp32 * tmptwo, dim=d_reduction_dims_for_gamma_beta, keepdim=False)

        # Convert outputs back to original dtype
        dx = dx.to(x.dtype)
        dgx = dgx.to(gx.dtype)

        return [dx, dgx, dbeta, dgamma]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, gx: torch.Tensor,
                gamma: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, alpha: float) -> List[torch.Tensor]:
        import kernel_gen_ops
        return kernel_gen_ops.deep_norm_grad(dy, x, gx, gamma, mean, rstd, alpha)