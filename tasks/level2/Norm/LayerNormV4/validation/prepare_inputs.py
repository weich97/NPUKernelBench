import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the LayerNormV4 operator's forward method.
    """
    input_shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    input_tensor = torch.rand(input_shape, device=device, dtype=dtype)

    return (input_tensor,)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters (normalized_shape, weight, bias, eps) for the model.
    """
    normalized_shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    # --- FIX START ---
    # Convert epsilon to float explicitly
    eps = float(param.get('epsilon', 1e-5)) 
    # --- FIX END ---
    
    # Handle optional weight and bias
    weight_type = param.get('weight_type', 'present')
    bias_type = param.get('bias_type', 'present')

    weight = None
    if weight_type == 'present':
        # Ensure weight is created on device if specified
        weight = torch.rand(normalized_shape, device=device, dtype=dtype)
    
    bias = None
    if bias_type == 'present':
        # Ensure bias is created on device if specified
        bias = torch.rand(normalized_shape, device=device, dtype=dtype)

    return [normalized_shape, weight, bias, eps]