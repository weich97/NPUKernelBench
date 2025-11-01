import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    x1_shape = eval(param.get('x1_shape', '[[1]]'))
    x2_shape = eval(param.get('x2_shape', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # 创建随机张量
    x1 = torch.rand(x1_shape, device=device, dtype=dtype)
    x2 = torch.rand(x2_shape, device=device, dtype=dtype)

    t1 = float(param.get('t1', 0.3))
    t2 = float(param.get('t2', 0.1))
    t3 = float(param.get('t3', 0.8))

    return x1, x2, t1, t2, t3


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MulSigmoid.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed

from framework.utils import check_precision

def custom_check_precision(param, outputs, outputs_new):
    return check_precision(outputs, outputs_new, max_abs_error=0.003, max_rel_error=0.003)