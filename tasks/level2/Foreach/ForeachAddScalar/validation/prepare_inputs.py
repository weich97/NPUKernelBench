import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # 创建随机张量列表
    x = []
    for shape in shape_list:
        x0 = torch.rand(shape, device=device, dtype=dtype)
        x.append(x0)
    if dtype_str == 'bfloat16':
        scalar_dtype = getattr(torch, 'float')
    else:
        scalar_dtype = getattr(torch, dtype_str)

    scalar_value = float(param.get('scalar', 1.0))
    alpha = torch.tensor([scalar_value], device=device, dtype=scalar_dtype)

    return x, alpha


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