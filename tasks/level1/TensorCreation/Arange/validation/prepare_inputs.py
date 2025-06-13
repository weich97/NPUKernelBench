import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    start = param.get('start', '[1]')
    end = param.get('end', '[1]')
    step = param.get('step', '[1]')

    dtype_str = param.get('dtype', 'float16')

    dtype = getattr(torch, dtype_str)

    start = torch.tensor(start, device=device, dtype=dtype)
    end = torch.tensor(end, device=device, dtype=dtype)
    step = torch.tensor(step, device=device, dtype=dtype)

    return (start, end, step)


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
