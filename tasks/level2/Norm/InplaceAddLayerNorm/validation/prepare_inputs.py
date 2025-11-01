import torch
from framework.utils import check_precision

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成模型的输入张量列表和标量。
    """
    shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    bias_type = param.get('bias_type', 'null')

    x1 = torch.rand(shape, device=device, dtype=dtype)
    x2 = torch.rand(shape, device=device, dtype=dtype)
    bias = None
    if bias_type == 'present':
        bias = torch.rand(shape, device=device, dtype=dtype)
    elif bias_type == 'broadcast':
        bias_shape = eval(param.get('normalized_shape', '[1]'))
        bias = torch.rand(bias_shape, device=device, dtype=dtype)

    return (x1, x2, bias)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for foreach_mul_scalar.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
  

    gamma = torch.rand(shape, device=device, dtype=dtype)
    beta = torch.rand(shape, device=device, dtype=dtype)

    epsilon = param.get('epsilon', 1e-5)
    additional_out = param.get('additional_out', False)
    
    return [gamma, beta, epsilon, additional_out]  # No special initialization inputs needed

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    if dtype_str == 'bfloat16' or dtype_str == 'float16':
        return check_precision(outputs, outputs_new, max_abs_error=0.01, max_rel_error=0.01)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.001, max_rel_error=0.001)