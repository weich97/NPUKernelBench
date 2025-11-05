import torch
import numpy as np
from typing import List, Tuple


def get_inputs(param, device=None) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
    """
    返回：
        scaled_grads : List[Tensor]  ← 一层列表
        found_inf    : Tensor(0/1, float)
        in_scale     : Tensor(scale, float)
    """
    input_shapes = eval(param.get('input_shape', '[[8, 2048]]'))
    dtype_str = param.get('dtype', 'float32')
    dtype = getattr(torch, dtype_str)
    scale_value = float(param.get('scale_value', 1.0))
    add_non_finite = bool(param.get('add_non_finite', False))

    scaled_grads = []
    for shape in input_shapes:
        g = (torch.rand(shape, device=device, dtype=dtype) * 0.01)
        if add_non_finite and g.numel():
            g.view(-1)[0] = float('nan') if np.random.rand() > 0.5 else float('inf')
        scaled_grads.append(g)

    found_inf = torch.tensor(0.0, dtype=torch.float, device=device)
    in_scale  = torch.tensor(scale_value, dtype=torch.float, device=device)

    return (scaled_grads, found_inf, in_scale)


def get_init_inputs(param, device=None) -> List:
    return []

def custom_check_precision(param, outputs, outputs_new):
    # Ensure inputs are in list format
    max_abs_error = max_rel_error = 0.01
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_accurate = True

    # Process each output pair
    for out, out_new in zip(outputs, outputs_new):
        if out_new.dtype == torch.bool:
            out = out.to(torch.int)
            out_new = out_new.to(torch.int)
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        all_abs_diff.append(abs_diff.reshape(-1))
        all_rel_diff.append(rel_diff.reshape(-1))

        # Check if within precision requirements
        if ((abs_diff > max_abs_error) & (rel_diff > max_rel_error)).any():
            is_accurate = False

    # Combine all differences
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)

    return (1 if is_accurate else 0), all_abs_diff, all_rel_diff