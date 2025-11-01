import torch
from framework.utils import check_precision

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    m = int(param.get('m', 1))
    n = int(param.get('n', 1))
    
    # 生成随机的实部和虚部（必须是 float32/float64）
    real = torch.rand(m, n, dtype=torch.float32, device=device)  # 实部
    imag = torch.rand(m, n, dtype=torch.float32, device=device)  # 虚部

    # 组合成复数矩阵
    matx = torch.complex(real, imag)  # dtype=torch.complex64
    
    # 生成随机的实部和虚部（必须是 float32/float64）
    real = torch.rand(m, n, dtype=torch.float32, device=device)  # 实部
    imag = torch.rand(m, n, dtype=torch.float32, device=device)  # 虚部

    # 组合成复数矩阵
    maty = torch.complex(real, imag)  # dtype=torch.complex64

    return matx, maty, m, n


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for ComplexMatDot.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed

def custom_check_precision(param, outputs, outputs_new):
    outputs = outputs.to('cpu')
    outputs_new = outputs_new.to('cpu')
    return check_precision(outputs, outputs_new, max_abs_error=0.00001, max_rel_error=0.00001)