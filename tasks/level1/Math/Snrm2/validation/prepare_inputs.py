import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the Snrm2 operator's forward method.
    """
    input_shape = eval(param.get('input_shape', '[1]')) 
    
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    
    x = torch.rand(input_shape, device=device, dtype=dtype) * 200 - 100 # Values from -100 to 100
    
    return (x,)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters (n, incx) for the Snrm2 model.
    """
    n = int(param.get('n', 1)) # Number of elements to consider
    incx = int(param.get('incx', 1)) # Stride
    
    return [n, incx]