import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the GroupNormSwishGrad operator's forward method.
    """
    x_shape = eval(param.get('x_shape', '[2, 32, 4, 4]')) # Example shape (N, C, H, W)
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    num_channels = x_shape[1]
    num_groups = param.get('num_groups', 8)
    
    # Inputs for GroupNormSwishGrad
    dy = (torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0) * 0.1 # Scale down random inputs to avoid overflow
                                                                           # For better stability during grad calculation
    # Mean and RSTD from GroupNorm forward pass
    # Shape of mean and rstd: (N, num_groups)
    mean_rstd_shape = (x_shape[0], num_groups)
    mean = torch.rand(mean_rstd_shape, device=device, dtype=dtype) * 0.1 # Small random values
    rstd = torch.rand(mean_rstd_shape, device=device, dtype=dtype) * 0.1 + 1e-3 # Ensure positive and not too large

    x = (torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0) * 0.1 # Original input to GroupNormSwish, scaled
    gamma = torch.rand(num_channels, device=device, dtype=dtype) * 0.1 + 0.5 # Gamma param, usually around 1
    beta = torch.rand(num_channels, device=device, dtype=dtype) * 0.1 # Beta param, usually around 0

    # Optional flags for dgamma/dbeta calculation. Convert 1/0 from CSV to True/False.
    dgamma_is_require = bool(int(param.get('dgamma_is_require', True)))
    dbeta_is_require = bool(int(param.get('dbeta_is_require', True)))

    return (dy, mean, rstd, x, gamma, beta, dgamma_is_require, dbeta_is_require)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the GroupNormSwishGrad model from DataFrame row.
    """
    num_groups = param.get('num_groups', 8)
    swish_scale = float(param.get('swish_scale', 1.0))

    return [num_groups, swish_scale]