import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    dims = eval(param.get('dims', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # Implementation note.
    value_str = param.get('value', '1.0')
    if dtype_str == 'bool':
        # Implementation note.
        value_value = (value_str.lower() == 'true')
    elif dtype_str in ['int8', 'int32', 'int64']:
        # Implementation note.
        value_value = int(value_str)
    else:
        # Implementation note.
        value_value = float(value_str)

    # Implementation note.
    value_tensor = torch.tensor(value_value, device=device, dtype=dtype)

    return [dims, value_tensor]


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for Fill.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed