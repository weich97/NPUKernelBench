import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))

    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)


    input1 = torch.randn(shape, device=device, dtype=dtype) * 2 - 1  # 生成 [0,1) 均匀分布，乘2变成 [0,2)，减1变成 [-1,1)
    input2 = torch.randn(shape, device=device, dtype=dtype) * 2 - 1

    return (input1, input2)


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
