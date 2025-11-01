import torch


def get_inputs(param, device=None):
    shape_list = eval(param.get('input_shape', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    x1 = []
    x2 = []
    for shape in shape_list:
        x = torch.randn(shape, device=device, dtype=dtype)
        y = torch.rand(shape, device=device, dtype=dtype) + 1

        x1.append(x)
        x2.append(y)

    return x1, x2


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