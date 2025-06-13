import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    scale = float(param.get('scale', 1.0))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # 创建随机张量列表
    inputs1 = []
    inputs2 = []
    for shape in shape_list:
        if dtype == torch.int32:
            # 整数类型使用randint
            x = torch.randint(-100, 100, shape, device=device, dtype=dtype)
            y = torch.randint(-100, 100, shape, device=device, dtype=dtype)
        else:
            # 浮点类型使用randn
            x = torch.randn(shape, device=device, dtype=dtype) * scale
            y = torch.randn(shape, device=device, dtype=dtype) * scale

        inputs1.append(x)
        inputs2.append(y)

    return inputs1, inputs2


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
