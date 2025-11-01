import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    self_shape = eval(param.get('self_shape', '[5, 3]'))  # Default shape [5,3]
    indices_shape = eval(param.get('indices_shape', '[2]'))  # Default shape [2]
    
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    indices_dtype = torch.int64  # index_select requires int64 indices
    
    # Generate random input tensor
    self_tensor = torch.randn(self_shape, device=device, dtype=dtype)
    
    # Generate indices - must be in valid range for the selected dimension
    # For index_select, indices must be 1D and values must be < dim_size
    dim_size = self_shape[0]  # default to first dimension
    if 'axis' in param:
        axis = int(param['axis'])
        dim_size = self_shape[axis]
    
    indices = torch.randint(0, dim_size, indices_shape, device=device, dtype=indices_dtype)
    
    # Axis is a scalar tensor indicating which dimension to select from
    axis = torch.tensor(param.get('axis', 0), device=device, dtype=torch.int64)

    return (self_tensor, axis, indices)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for GatherV3.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed