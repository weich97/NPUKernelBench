import torch
import ast


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    # Parse shapes
    x_shapes = ast.literal_eval(param.get('x_shapes', '[[1, 1]]'))
    gmm_shapes = ast.literal_eval(param.get('gmm_shapes', '[[1, 1]]'))  # weight shapes
    bias_shapes = ast.literal_eval(param.get('bias_shapes', '[]'))

    # Parse group_list_data directly
    group_list_data = param.get('group_list_data', '')
    group_list = None
    if group_list_data:
        group_list = ast.literal_eval(group_list_data)

    # Parse other parameters
    split_item = int(param.get('split_item', '0'))
    transpose_weight = param.get('transpose_weight', 'false').lower() == 'true'
    transpose_x = param.get('transpose_x', 'false').lower() == 'true'

    # Get dtype
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # Create x tensors
    if len(x_shapes) == 1 and len(x_shapes[0]) == 2:
        # Single tensor case
        x = torch.randn(x_shapes[0], device=device, dtype=dtype)
    else:
        # Multiple tensors case
        x = [torch.randn(shape, device=device, dtype=dtype) for shape in x_shapes]

    # Create weight tensors
    weight = [torch.randn(shape, device=device, dtype=dtype) for shape in gmm_shapes]

    # Create bias tensors if provided
    bias = None
    if bias_shapes:
        bias = [torch.randn(shape, device=device, dtype=dtype) for shape in bias_shapes]

    return (x, weight, bias, group_list, split_item, transpose_weight, transpose_x)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for grouped matmul.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed