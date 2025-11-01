from typing import List, Optional

import torch
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):

    def __init__(self, normalized_shape: List[int], weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], eps: float):
        super(Model, self).__init__()
        self.normalized_shape = normalized_shape
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        # Convert all inputs to float32 for intermediate calculations
        input_fp32 = input_tensor.to(torch.float32)
        weight_fp32 = self.weight.to(torch.float32) if self.weight is not None else None
        bias_fp32 = self.bias.to(torch.float32) if self.bias is not None else None

        # Calculate reduction dimensions based on normalized_shape
        input_dim = input_fp32.dim()
        normalized_dim = len(self.normalized_shape)
        reduction_dims = tuple(range(input_dim - normalized_dim, input_dim))

        # Calculate mean
        mean = input_fp32.mean(dim=reduction_dims, keepdim=True)

        # Calculate variance
        variance = (input_fp32 - mean).pow(2).mean(dim=reduction_dims, keepdim=True)

        # Calculate rstd (reciprocal standard deviation)
        rstd = torch.rsqrt(variance + self.eps)

        # Normalize input
        out = (input_fp32 - mean) * rstd

        # Apply weight and bias if provided
        if weight_fp32 is not None:
            out = out * weight_fp32
        if bias_fp32 is not None:
            out = out + bias_fp32

        # Convert outputs back to original dtype
        out = out.to(input_tensor.dtype)
        mean = mean.to(input_tensor.dtype)
        rstd = rstd.to(input_tensor.dtype)

        return [out, mean, rstd]


class ModelNew(nn.Module):
    def __init__(self, normalized_shape: List[int], weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], eps: float):
        super(ModelNew, self).__init__()
        # LayerNormV4 requires normalized_shape for its C++ implementation
        self.normalized_shape_list = normalized_shape # Store as list
        self.weight = weight
        self.bias = bias
        self.eps = eps

    def forward(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        # Pass normalized_shape as a tuple to the C++ op if it expects IntArrayRef
        # Kernel ops might internally convert list to IntArrayRef
        return kernel_gen_ops.layer_norm_v4(input_tensor, tuple(self.normalized_shape_list), self.weight, self.bias, self.eps)