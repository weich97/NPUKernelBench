import torch
from typing import List, Tuple


def get_inputs(param, device=None) -> Tuple[List[torch.Tensor]]:
    """
    Generate input tensors for the Erf model based on parameters from DataFrame row.
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    scale = float(param.get('scale', 1.0))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # Create random tensor list
    inputs = []
    for shape in shape_list:
        # Erf function domain is [-inf, inf], but for precision,
        # it's good to generate inputs within a reasonable range (e.g., around 0)
        # to avoid extreme values where erf approaches +/-1 quickly.
        # Let's generate values around 0, scaled.
        x = (torch.rand(shape, device=device, dtype=dtype) * 2.0 - 1.0) * scale # Scale to [-scale, scale] range
        inputs.append(x)

    # Return a tuple containing the list of tensors as its single element.
    # This ensures that when unpacked by `*inputs_copy`, `inputs` in forward
    # correctly receives the list of tensors.
    return (inputs,)


def get_init_inputs(param, device=None) -> List:
    """
    Extract initialization parameters for the Erf model from DataFrame row.
    No special initialization needed for Erf.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed