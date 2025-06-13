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
        
        # Determine the normalization dimension (D) for the mean/variance calculations
        # Assuming last dimension is normalized, consistent with DeepNorm forward pass
        # D should be the size of the innermost normalized dimension.
        D = float(x.shape[-1]) # Or gamma.shape[-1] if gamma is 1D

        # Calculate intermediate terms
        tmpone = dy * gamma
        tmptwo = alpha * x + gx - mean

        # Calculate d_var and d_mean (sum over the normalized dimension)
        # Use keepdim=True for sum to maintain dimensions for broadcasting
        # The sum should be over the dimension(s) that were normalized in the forward pass.
        # Assuming the last dimension was normalized, so sum over -1.
        d_var = torch.sum(-0.5 * tmpone * tmptwo * rstd.pow(3), dim=-1, keepdim=True)
        d_mean = torch.sum(-1.0 * tmpone * rstd, dim=-1, keepdim=True)

        # Calculate d_gx
        # Note: The formula for d_gx_i seems to miss a term relating to alpha*x.
        # It's usually like: d_gx = tmpone * rstd + 2/D * d_var * tmptwo + 1/D * d_mean
        # This matches common LayerNorm-like backward passes.
        dgx = tmpone * rstd + (2.0 / D) * d_var * tmptwo + (1.0 / D) * d_mean

        # Calculate d_x
        # d_x_i = alpha * d_gx_i (This is a simplified form, usually derived from chain rule)
        # Assuming the formula provided is accurate for this specific DeepNormGrad.
        dx = alpha * dgx # This aligns with `d_x_i = alpha * {gx}_i` if gx is meant as a placeholder for dgx.
                         # If it literally means the input gx, then the formula is suspicious for a grad.
                         # Assuming `dgx_i` is the gradient we just calculated for `gx`.
        # Re-interpreting: the formula d_x_i = alpha * {gx}_i seems like a typo or specific simplification.
        # Standard chain rule for `x_add = x * alpha + gx` would imply:
        # d(x_add)/d(x) = alpha, d(x_add)/d(gx) = 1
        # So dx = dgx_add * alpha, dgx = dgx_add * 1.
        # Where dgx_add is the gradient of `x_add` after norm.
        # The provided d_gx is `dgx_add`. So dx = dgx_add * alpha = dgx * alpha.
        # This seems consistent.

        # Calculate d_beta (sum over all dimensions except normalized_shape)
        # d_beta should be a sum over all dimensions that were *not* normalized.
        # If gamma/beta are 1D (e.g., shape D), and x is (B,L,D), then sum over (0,1).
        # This is `dy_i` summed for each element of beta/gamma.
        dbeta_reduction_dims = tuple(range(dy.dim() - gamma.dim()))
        dbeta = torch.sum(dy, dim=dbeta_reduction_dims, keepdim=False)

        # Calculate d_gamma (sum over all dimensions except normalized_shape)
        # d_gamma = sum(dy_i * rstd * tmptwo_i)
        dgamma = torch.sum(dy * rstd * tmptwo, dim=dbeta_reduction_dims, keepdim=False) # Reuse dbeta_reduction_dims

        return [dx, dgx, dbeta, dgamma]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, dy: torch.Tensor, x: torch.Tensor, gx: torch.Tensor,
                gamma: torch.Tensor, mean: torch.Tensor, rstd: torch.Tensor, alpha: float) -> List[torch.Tensor]:
        import kernel_gen_ops
        return kernel_gen_ops.deep_norm_grad(dy, x, gx, gamma, mean, rstd, alpha)