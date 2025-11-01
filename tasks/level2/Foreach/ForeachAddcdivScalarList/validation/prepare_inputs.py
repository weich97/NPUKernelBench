import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    scalars_values = eval(param.get('scalars', '[1.0]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # 创建随机张量列表
    x1 = []
    x2 = []
    x3 = []
    for shape in shape_list:

        # 浮点类型使用randn
        x = torch.randn(shape, device=device, dtype=dtype)
        y = torch.randn(shape, device=device, dtype=dtype)
        z = torch.rand(shape, device=device, dtype=dtype) + 1

        x1.append(x)
        x2.append(y)
        x3.append(z)

    scalars = torch.tensor(scalars_values, device=device, dtype=dtype)

    return x1, x2, x3, scalars


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for ForeachAddcdivList.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed

from framework.utils import check_precision

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    if dtype_str == 'bfloat16':
        return check_precision(outputs, outputs_new, max_abs_error=0.02, max_rel_error=0.02)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)