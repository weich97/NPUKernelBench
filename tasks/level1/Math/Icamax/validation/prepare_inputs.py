import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the Icamax operator's forward method.
    """
    input_shape = eval(param.get('input_shape', '[1]')) 
    dtype_str = param.get('dtype', 'complex64') # Default to complex64
    
    # Map string dtype to PyTorch complex dtype
    if dtype_str == 'complex64':
        torch_dtype = torch.complex64
        np_real_dtype = np.float32
    elif dtype_str == 'complex128':
        torch_dtype = torch.complex128
        np_real_dtype = np.float64
    else:
        raise ValueError(f"Unsupported complex dtype: {dtype_str}")

    # Generate real and imaginary parts separately as floats
    x_r = np.random.uniform(-1, 1, input_shape).astype(np_real_dtype)
    x_i = np.random.uniform(-1, 1, input_shape).astype(np_real_dtype)
    
    # Create the complex numpy array, then convert to torch.Tensor
    x_np = (x_r + 1j * x_i)
    x = torch.from_numpy(x_np).to(device=device, dtype=torch_dtype)
    
    return (x,)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters (n, incx) for the Icamax model.
    """
    n = int(param.get('n', 1)) 
    incx = int(param.get('incx', 1)) 
    
    return [n, incx]