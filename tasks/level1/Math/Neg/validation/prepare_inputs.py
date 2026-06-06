import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    if dtype == torch.int32 or dtype == torch.int8:
        # Implementation note.
        x = torch.randint(-100, 100, shape, device=device, dtype=dtype)
    else:
        # Implementation note.
        x = torch.randn(shape, device=device, dtype=dtype)

    return (x,)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for Neg.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed

