import torch
from framework.utils import check_precision


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    mul0input0 = torch.rand(shape, device=device, dtype=dtype)
    mul0input1 = torch.rand(shape, device=device, dtype=dtype)
    mul1input0 = torch.tensor(1.0, device=device, dtype=dtype)
    addy = torch.tensor(1.0, device=device, dtype=dtype)
    gamma = torch.rand(1, shape[1], device=device, dtype=dtype)
    beta = torch.rand(1, shape[1], device=device, dtype=dtype)
    
    return (mul0input0, mul0input1, mul1input0, addy, gamma, beta)


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

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    if dtype_str == 'bfloat16' or dtype_str == 'float16':
        return check_precision(outputs, outputs_new, max_abs_error=0.005, max_rel_error=0.005)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.002, max_rel_error=0.002)