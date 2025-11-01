import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    alpha = float(param.get('alpha', 2.0))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    # input = torch.randn(shape, device=device, dtype=dtype)

    input = torch.ones(shape, device=device, dtype=dtype)

    return (input, alpha)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    n    = np.int64(param.get('n', 1))
    incx = np.int64(1)
    return (n, incx)