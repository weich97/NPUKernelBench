import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    

    input = (torch.rand(shape, dtype=dtype, device=device) * 9) + 1

    clip_value_min = (torch.rand(1, dtype=dtype, device=device) * 2) + 1
    clip_value_max = (torch.rand(1, dtype=dtype, device=device) * 6) + 4

    return (input, clip_value_min, clip_value_max)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed
