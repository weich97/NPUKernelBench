from typing import List, Tuple, Optional

import torch
import torch.nn as nn
# Removed: import torch.nn.functional as F  # Not needed if implementing manually
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self, num_features: int, eps: float, momentum: float, affine: bool):
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

    def forward(self, input_tensor: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                running_mean: torch.Tensor, running_var: torch.Tensor, training: bool) -> List[torch.Tensor]:
        
        # Determine the dimensions over which to normalize.
        # For BatchNorm, normalization happens over all dimensions EXCEPT the feature dimension (C).
        # Assuming input_tensor is (N, C, H, W) or (N, C, D, H, W) or (N, C).
        # The feature dimension is at index 1.
        reduction_dims = [0] + list(range(2, input_tensor.dim()))
        # E.g., for (N, C, H, W), reduction_dims will be [0, 2, 3] (N, H, W)

        # 1. Calculate Batch Mean (E[x])
        batch_mean = input_tensor.mean(dim=reduction_dims, keepdim=True)

        # 2. Calculate Batch Variance (Var[x])
        batch_variance = (input_tensor - batch_mean).pow(2).mean(dim=reduction_dims, keepdim=True)

        # 3. Calculate Inverse Standard Deviation (1 / sqrt(Var[x] + ε))
        # This is saveInvstd.
        save_invstd = torch.rsqrt(batch_variance + self.eps)

        # 4. Normalize the input using batch statistics
        normalized_input = (input_tensor - batch_mean) * save_invstd

        # 5. Apply affine transformation (gamma * normalized_x + beta)
        output = normalized_input
        if weight is not None:
            output = output * weight.view(1, -1, *([1] * (input_tensor.dim() - 2))) # Reshape weight for broadcasting
        if bias is not None:
            output = output + bias.view(1, -1, *([1] * (input_tensor.dim() - 2)))   # Reshape bias for broadcasting

        # 6. Update running_mean and running_var if in training mode
        if training:
            with torch.no_grad(): # Ensure these updates don't create graph nodes
                # Reshape batch_mean/variance for correct broadcasting with running_mean/var
                # E.g., if batch_mean is (1, C, 1, 1), reshape to (C,) for update
                running_mean.copy_(
                    (1 - self.momentum) * running_mean + self.momentum * batch_mean.squeeze()
                )
                running_var.copy_(
                    (1 - self.momentum) * running_var + self.momentum * batch_variance.squeeze()
                )
        
        # The outputs for aclnnBatchNormGetWorkspaceSize are output, saveMean, saveInvstd
        # Here, saveMean is the calculated batch_mean.
        return [output, batch_mean.squeeze(), save_invstd.squeeze()]


class ModelNew(nn.Module):
    def __init__(self, num_features: int, eps: float, momentum: float, affine: bool):
        super(ModelNew, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine

    def forward(self, input_tensor: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor],
                running_mean: torch.Tensor, running_var: torch.Tensor, training: bool) -> List[torch.Tensor]:
        
        return kernel_gen_ops.batch_norm_v3(
            input_tensor,
            weight,
            bias,
            running_mean,
            running_var,
            training,
            self.momentum,
            self.eps
        )