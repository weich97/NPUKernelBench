import torch


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
    dtype = getattr(torch, dtype_str)

    x = torch.rand(shape, device=device, dtype=dtype)
    return (x,)


def get_init_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
    return []