import torch
import numpy as np

def get_inputs(param, device=None):
    """
    Generate input tensors for the AddLayerNormGrad operator's forward method.
    """
    input_shape = eval(param.get('input_shape', '[3, 1, 4]'))
    normalized_shape = eval(param.get('normalized_shape', '[4]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    dsum_type = param.get('dsum_type', 'None')

    dy = torch.rand(input_shape, device=device, dtype=dtype)
    x1 = torch.rand(input_shape, device=device, dtype=dtype)
    x2 = torch.rand(input_shape, device=device, dtype=dtype)

    x = x1.to(torch.float32) + x2.to(torch.float32)
    dims_to_normalize = tuple(range(len(input_shape) - len(normalized_shape), len(input_shape)))
    mean = torch.mean(x, dim=dims_to_normalize, keepdim=True)
    var= torch.mean(torch.pow(x - mean, 2), dim=dims_to_normalize, keepdim=True)
    rstd = torch.rsqrt(var + 1e-5)

    gamma = torch.rand(normalized_shape, device=device, dtype=dtype)

    dsum = None
    if dsum_type == 'present':
        dsum = torch.rand(input_shape, device=device, dtype=dtype)

    return (dy, x1, x2, rstd, mean, gamma, dsum)

def get_init_inputs(param, device=None):
    normalized_shape = eval(param.get('normalized_shape', '[4]'))
    return [normalized_shape]

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float16:
        RTOL_GENERAL = 1 / 512
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 256 + 1 / 16384
    elif dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048 + 1 / 16384
    elif dtype == torch.int32:
        RTOL_GENERAL = 1

    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1
    for out, out_new in zip(outputs, outputs_new):
        # 计算绝对差值、相对误差
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        all_abs_diff.append(abs_diff.view(-1))
        all_rel_diff.append(rel_diff.view(-1))

        # 计算容忍度阈值
        tolerance = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out))

        # 找出差异大于容忍度的位置
        error_mask = abs_diff > tolerance

        # 检查是否有任何元素的差异超过了容忍度
        if torch.any(error_mask):
            is_pass = 0


    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)

    return is_pass, all_abs_diff, all_rel_diff