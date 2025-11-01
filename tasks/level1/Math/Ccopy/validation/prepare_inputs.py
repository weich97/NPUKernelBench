import torch
import numpy as np
from framework.utils import check_precision

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    input_shape = eval(param.get('input_shape', '[1]'))
    real_part = torch.rand(input_shape, dtype=torch.float32, device=device)
    imag_part = torch.rand(input_shape, dtype=torch.float32, device=device)
    x = torch.complex(real_part, imag_part)
    return [x]


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters (n, incx) for the Sasum model.
    """

    n = int(param.get('n', 1))  # Number of elements to consider
    incx = 1  # Stride
    incy = 1  # Stride
    return [n, incx, incy]

def custom_check_precision(param, outputs, outputs_new):
    outputs = outputs.to('cpu')
    outputs_new = outputs_new.to('cpu')
    return check_precision(outputs, outputs_new, max_abs_error=0.00001, max_rel_error=0.00001)
