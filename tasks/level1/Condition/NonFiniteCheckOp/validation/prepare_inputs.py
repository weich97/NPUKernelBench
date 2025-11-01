import torch
import numpy as np
from typing import List, Tuple

def get_inputs(param, device=None):
    """
    Generate input tensors for the NonFiniteCheck operator.
    """
    input_shape = eval(param.get('input_shape', '[[1]]'))  # Expecting list of lists
    dtype_str = param.get('dtype', 'float32')

    # Get the PyTorch dtype
    torch_dtype = getattr(torch, dtype_str)

    contains_non_finite = param.get('contains_non_finite', False)

    x = []
    for shape in input_shape:
        np_array = np.random.rand(*shape)
        t = torch.from_numpy(np_array).to(device=device, dtype=torch_dtype)
        x.append(t)

    if contains_non_finite:
        for t in x:
            num_elements = t.numel()
            if num_elements > 0:
                choice = np.random.choice(['inf', 'nan'])
                if choice == 'inf':
                    t.view(-1)[np.random.randint(num_elements)] = float('inf')
                else:
                    t.view(-1)[np.random.randint(num_elements)] = float('nan')
                if num_elements > 1:
                    t.view(-1)[np.random.randint(num_elements)] = float('-inf')

    return (x,)

def get_init_inputs(param, device=None):
    """
    NonFiniteCheck Model has no initialization parameters.
    """
    return []