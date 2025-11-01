import torch
import numpy as np
from typing import List, Tuple


def get_inputs(param, device=None) -> Tuple[List[torch.Tensor], torch.Tensor]:
    input_shapes = eval(param.get('input_shape', '[[8, 2048]]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    exponent = param.get('scale_value', 2.0)

    grads = [torch.randn(shape, device=device, dtype=dtype) for shape in input_shapes]

    exp_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    exp_tensor = torch.tensor(exponent, dtype=exp_dtype, device=device)

    return grads, exp_tensor


def get_init_inputs(param, device=None) -> List:
    
    return []

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