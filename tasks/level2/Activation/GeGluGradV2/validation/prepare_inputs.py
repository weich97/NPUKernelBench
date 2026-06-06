# prepare_inputs.py
import torch
import ast  # Implementation note.
import torch.nn.functional as F

def get_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.
        Reference implementation detail.

    Returns:
        Reference implementation detail.
        Reference implementation detail.
    """
    shape_str = param.get('input_shape', '[1, 2]')
    shape = ast.literal_eval(shape_str)

    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    dim = int(param.get('dim', -1))

    # Implementation note.
    approximate = int(param.get('approximate', 0))

    activateLeft = bool(param.get('activateLeft', True))

    dy = torch.rand(shape, device=device, dtype=dtype)

    x_shape = list(shape)
    # Implementation note.
    # Implementation note.
    if dim == -1:
        x_shape[-1] = x_shape[-1] * 2
    elif dim < len(x_shape):
        x_shape[dim] = x_shape[dim] * 2
    else:
        # Implementation note.
        raise ValueError(f"Invalid dim: {dim} for shape {shape}")

    # Create x tensor with requires_grad=True from the start
    x = torch.rand(x_shape, device=device, dtype=dtype, requires_grad=True)

    return (dy, x, dim, approximate, activateLeft)


def get_init_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
    return []