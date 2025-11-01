import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the DeepNorm operator.
    """
    input_shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    x = torch.rand(input_shape, device=device, dtype=dtype)
    gx = torch.rand(input_shape, device=device, dtype=dtype) # gx has same shape as x

    return (x, gx)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters (beta, gamma, alpha, epsilon) for the model.
    """
    normalized_shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    alpha = param.get('alpha', 0.3)
    epsilon = param.get('epsilon', 1e-6)

    beta = torch.rand(normalized_shape, device=device, dtype=dtype)
    gamma = torch.rand(normalized_shape, device=device, dtype=dtype)
    
    return [beta, gamma, alpha, epsilon]