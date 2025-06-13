import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    input_tensor = torch.randn(shape, device=device, dtype=dtype)
    mask = torch.randint(0, 2, shape, device=device, dtype=torch.bool)

    return (input_tensor, mask)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for MaskSelectV3.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed