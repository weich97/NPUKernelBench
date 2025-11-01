import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    add_0_input0 = torch.randn(shape, device=device, dtype=dtype)* 0.4 - 0.2
    add_0_input1 = torch.randn((1, 1, shape[2]), device=device, dtype=dtype)* 0.4 - 0.2
    mult_0_input1 = torch.randn((1,), device=device, dtype=dtype)

    mult_1_input1 = torch.randn((shape[0], shape[1], 1), device=device, dtype=dtype)* 0.4 - 0.2
    mult_2_input1 = torch.randn(shape, device=device, dtype=dtype) * 0.4 - 0.2

    return (add_0_input0, add_0_input1, mult_0_input1, mult_1_input1, mult_2_input1)


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
        return check_precision(outputs, outputs_new, max_abs_error=0.01, max_rel_error=0.01)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.01, max_rel_error=0.01)
