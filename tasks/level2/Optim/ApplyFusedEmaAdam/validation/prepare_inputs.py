import torch
from framework.utils import check_precision

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    step = torch.tensor([param.get('step', 1)], device=device, dtype=torch.int64)

    # 生成输入张量
    grad = torch.randn(shape, device=device, dtype=dtype)
    varRef = torch.randn(shape, device=device, dtype=dtype)
    mRef = torch.zeros(shape, device=device, dtype=dtype)  # 初始化为0
    vRef = torch.zeros(shape, device=device, dtype=dtype)  # 初始化为0
    sRef = torch.zeros(shape, device=device, dtype=dtype)  # 初始化为0

    return grad, varRef, mRef, vRef, sRef, step


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        dict: Model initialization parameters
    """
    # 从CSV读取初始化参数，如果不存在则使用默认值
    lr = param.get('lr', 0.001)
    emaDecay = param.get('emaDecay', 0.999)
    beta1 = param.get('beta1', 0.9)
    beta2 = param.get('beta2', 0.999)
    eps = param.get('eps', 1e-8)
    mode = param.get('mode', 0)
    biasCorrection = bool(param.get('biasCorrection', 'True'))
    weightDecay = param.get('weightDecay', 0.0)
    return lr, emaDecay, beta1, beta2, eps, mode, biasCorrection, weightDecay

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    if dtype_str == 'bfloat16':
        return check_precision(outputs, outputs_new, max_abs_error=0.01, max_rel_error=0.01)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.001, max_rel_error=0.001)