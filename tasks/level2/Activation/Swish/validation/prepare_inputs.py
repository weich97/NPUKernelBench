import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    input1 = torch.randn(shape, device=device, dtype=dtype)
    scale = param.get('scale', 1.0)

    # input1 = (4 - 1) * torch.rand(shape, device=device, dtype=dtype) + 1
    # input1 = torch.empty(shape, device=device, dtype=dtype).uniform_(1, 4)

    return (input1, scale)


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
