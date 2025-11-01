import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape_x1 = eval(param.get('input_shape_x1', '[[1]]'))
    shape_x2 = eval(param.get('input_shape_x2', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    if dtype == torch.int32:
        # 整数类型使用randint
        x1 = torch.randint(-100, 100, shape_x1, device=device, dtype=dtype)
        x2 = torch.randint(-100, 100, shape_x2, device=device, dtype=dtype)
    else:
        # 浮点类型使用randn
        x1 = torch.randn(shape_x1, device=device, dtype=dtype)
        x2 = torch.randn(shape_x2, device=device, dtype=dtype)

    return x1, x2


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for Less.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed
