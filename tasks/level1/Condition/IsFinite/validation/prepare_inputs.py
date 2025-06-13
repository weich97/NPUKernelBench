import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the IsFinite operator.
    """
    input_shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float32')
    
    # Get the PyTorch dtype
    torch_dtype = getattr(torch, dtype_str)
    
    # Convert PyTorch dtype to NumPy dtype for numpy.astype()
    numpy_dtype = torch.empty(0, dtype=torch_dtype).numpy().dtype
    
    # Create a tensor with a mix of finite numbers, infinities, and NaNs
    np_array = np.random.rand(*input_shape).astype(numpy_dtype)
    
    # Inject infinity and NaN values into the array to test finiteness
    num_elements = np_array.size
    
    # Inject positive infinity
    if num_elements > 0:
        np_array.flat[np.random.randint(num_elements)] = np.inf
    
    # Inject negative infinity
    if num_elements > 1:
        np_array.flat[np.random.randint(num_elements)] = -np.inf
        
    # Inject NaN
    if num_elements > 2:
        np_array.flat[np.random.randint(num_elements)] = np.nan
        
    # Ensure some zeros are present if not naturally occurring
    if num_elements > 3:
        np_array.flat[np.random.randint(num_elements)] = 0.0

    x = torch.from_numpy(np_array).to(device=device, dtype=torch_dtype)
    
    return (x,)

def get_init_inputs(param, device=None):
    """
    IsFinite Model has no initialization parameters.
    """
    return []