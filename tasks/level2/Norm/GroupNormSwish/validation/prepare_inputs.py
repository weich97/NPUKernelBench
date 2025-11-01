import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the GroupNormSwish operator's forward method.
    """
    x_shape = eval(param.get('x_shape', '[100, 32]'))
    gamma_shape = eval(param.get('gamma_shape', '[32]'))
    beta_shape = eval(param.get('beta_shape', '[32]'))
    
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)

    x = torch.rand(x_shape, device=device, dtype=dtype) * 2.0 - 1.0 # Uniform in [-1, 1]
    gamma = torch.rand(gamma_shape, device=device, dtype=dtype)
    beta = torch.rand(beta_shape, device=device, dtype=dtype)

    return (x, gamma, beta)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the GroupNormSwish model from DataFrame row.
    """
    # num_channels (C) is the second dimension of x_shape, or the size of gamma/beta
    num_channels = eval(param.get('x_shape', '[100, 32]'))[1] # C = x_shape[1]
    num_groups = param.get('num_groups', 8)
    eps = float(param.get('eps', 1e-5))
    activate_swish = bool(param.get('activate_swish', False)) # Ensure bool conversion
    swish_scale = float(param.get('swish_scale', 1.0))

    return [num_channels, num_groups, eps, activate_swish, swish_scale]