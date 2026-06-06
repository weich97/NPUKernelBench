import torch

from framework.utils import check_precision


def get_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
    shape = eval(param.get('input_shape', '[1, 2]'))  # Implementation note.
    dtype_str = param.get('dtype', 'float16')
    dim = param.get('dim', -1)
    
    dtype = getattr(torch, dtype_str)
    x = torch.rand(shape, device=device, dtype=dtype)
    return (x, dim)


def get_init_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
    return []


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float32:
        return check_precision(outputs, outputs_new, max_abs_error=0.00001, max_rel_error=0.00001)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.01, max_rel_error=0.01)
