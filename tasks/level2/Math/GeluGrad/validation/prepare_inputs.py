import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # 创建dy输入
    dy = torch.rand(shape, device=device, dtype=dtype)

    # 创建x输入
    x = torch.rand(shape, device=device, dtype=dtype)

    return [dy, x]


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for gelu_grad.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed