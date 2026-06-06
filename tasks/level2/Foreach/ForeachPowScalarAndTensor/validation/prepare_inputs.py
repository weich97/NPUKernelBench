import torch
from typing import List, Tuple


def get_inputs(param, device=None) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.
                      Expected keys: 'input_shapes_str' (e.g., '[[10, 20], [5, 5]]'),
                      'weight_scalar', 'dtype'.
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """

    shape_list = eval(param.get('input_shape', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    raw = param.get('weight_scalar', 0.5)

    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        weight_scalar = float(raw)          # Python float
    else:  # int32
        weight_scalar = int(raw)            # Python int64
    
    # Create list of tensors
    x_list = []
    for shape in shape_list:
        x_list.append(torch.rand(shape, device=device, dtype=dtype))

    return (x_list, weight_scalar)


def get_init_inputs(param, device=None) -> list:
    return []