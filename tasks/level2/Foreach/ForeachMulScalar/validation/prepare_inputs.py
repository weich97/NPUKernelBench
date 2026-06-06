import torch


def get_inputs(param, device=None):
    """
    Reference implementation detail.
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    scalar_value = float(param.get('scalar', '1.0'))  # Implementation note.
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    inputs = []
    for shape in shape_list:
        if dtype == torch.int32:
            # Implementation note.
            x = torch.randint(-100, 100, shape, device=device, dtype=dtype)
        else:
            # Implementation note.
            x = torch.randn(shape, device=device, dtype=dtype)
        inputs.append(x)

    # Implementation note.
    # Implementation note.
    # Implementation note.
    if dtype == torch.bfloat16:
        scalar = torch.tensor(scalar_value, device=device, dtype=torch.float)
    elif dtype == torch.int32:
        scalar = torch.tensor(int(scalar_value), device=device, dtype=torch.int32)
    else:  # Implementation note.
        scalar = torch.tensor(scalar_value, device=device, dtype=dtype)

    return inputs, scalar


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for foreach_mul_scalar.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed