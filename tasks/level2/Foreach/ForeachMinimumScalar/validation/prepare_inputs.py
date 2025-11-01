import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    scalar_value = float(param.get('scalar', 1.0))

    # 创建随机张量列表
    inputs = []
    for shape in shape_list:
        if dtype == torch.int32:
            # 整数类型使用randint
            x = torch.randint(-100, 100, shape, device=device, dtype=dtype)
        else:
            # 浮点类型使用randn
            x = torch.randn(shape, device=device, dtype=dtype)

        inputs.append(x)
        
    if dtype_str == 'bfloat16':
        scalar_dtype = getattr(torch, 'float')
    else:
        scalar_dtype = getattr(torch, dtype_str)

    scalar = torch.tensor([scalar_value], device=device, dtype=scalar_dtype)

    return inputs, scalar


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

from framework.utils import check_precision

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    if dtype_str == 'bfloat16':
        return check_precision(outputs, outputs_new, max_abs_error=0.005, max_rel_error=0.005)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.0001, max_rel_error=0.0001)