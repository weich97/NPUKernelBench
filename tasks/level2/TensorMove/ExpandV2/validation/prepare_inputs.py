import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape1 = eval(param.get('shape1', '[1]'))
    shape2 = eval(param.get('shape2', '[2]'))

    dtype_str = param.get('dtype', 'int64')
    dtype = getattr(torch, dtype_str)
    
    input = torch.randint(0, 10, shape1, device=device, dtype=dtype)
    return (input, shape2)


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
