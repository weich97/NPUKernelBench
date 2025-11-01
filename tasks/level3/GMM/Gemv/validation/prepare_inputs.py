import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    m = param.get('m')
    n = param.get('n')
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    a = torch.rand([m, n], device=device, dtype=dtype) * 2 - 1
    x = torch.rand([n], device=device, dtype=dtype) * 2 - 1
    y = torch.rand([m], device=device, dtype=dtype) * 2 - 1
    alpha = torch.rand(1, device=device, dtype=torch.float32) * 2 - 1
    beta = torch.rand(1, device=device, dtype=torch.float32) * 2 - 1

    return (a, x, y, alpha, beta)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float16:
        RTOL_GENERAL = 1 / 256
        RTOL_OVER_THRESHOLD = 1 / 128
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 128
        RTOL_OVER_THRESHOLD = 1 / 64
    elif dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048
        RTOL_OVER_THRESHOLD = 1 / 512

    compute_num = param.get('m')
    COMPUTE_NUM_THRESHOLD = 2048

    outputs = outputs.to(torch.float32)
    outputs_new = outputs_new.to(torch.float32)

    # 根据 compute_num 的值选择相对容忍度 (rtol)
    rtol = RTOL_GENERAL if compute_num < COMPUTE_NUM_THRESHOLD else RTOL_OVER_THRESHOLD

    # 1. 计算绝对差值
    abs_diff = torch.abs(outputs - outputs_new)

    # 2. 计算容忍度阈值
    tolerance = rtol * torch.maximum(torch.tensor(1.0, device=outputs.device), torch.abs(outputs))

    # 3. 找出差异大于容忍度的位置
    error_mask = abs_diff > tolerance

    # 检查是否有任何元素的差异超过了容忍度
    is_pass = 1 if not torch.any(error_mask) else 0

    # 计算相对误差时，为防止分母为零，加上一个极小值 epsilon
    relative_diff = abs_diff / (torch.abs(outputs) + 1e-7)

    return is_pass, abs_diff, relative_diff