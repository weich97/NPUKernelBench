import torch
import numpy as np
from typing import List, Tuple


def get_inputs(param, device=None):
    """
    返回：
        x         : Tensor         输入张量（不再是 List）
        scale     : Tensor(float)  量化 scale
        offset    : Tensor(float)  量化 offset
        quantMode : str            "static" or "dynamic"
    """
    input_shape = eval(param.get('input_shape', '[8, 2048]'))  # 单个 shape
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    scale_value = float(param.get('scale_value', 1.0))
    offset_value = float(param.get('offset_value', 0.0))
    approximate = param.get('approximate', 'tanh')
    quant_mode = param.get('quant_mode', 'static')

    x = torch.rand(input_shape, device=device, dtype=dtype) * 2 - 1

    scale_tensor = torch.tensor(scale_value, dtype=dtype, device=device)
    offset_tensor = torch.tensor(offset_value, dtype=dtype, device=device)

    return x, scale_tensor, offset_tensor, approximate, quant_mode


def get_init_inputs(param, device=None) -> List:
    return []

def custom_check_precision(param, outputs, outputs_new):
    is_pass = 1
    LOSS = 1e-3
    all_abs_diff, all_rel_diff = [], []
    for out, out_new in zip(outputs, outputs_new):
        result = torch.abs(out - out_new)

        # Get maximum of absolute values for relative error denominator
        deno = torch.maximum(torch.abs(out), torch.abs(out_new))

        # Calculate absolute error check
        result_atol = result

        # Calculate relative error check
        result_rtol = result / (deno + 1e-7)

        # Count failures where error exceeds tolerance
        if torch.sum(result_rtol > LOSS) > out.numel() * LOSS and \
        torch.sum(result_atol > LOSS) > out.numel() * LOSS:
            is_pass = 0
        all_abs_diff.append(result_atol.flatten())
        all_rel_diff.append(result_rtol.flatten())
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff  