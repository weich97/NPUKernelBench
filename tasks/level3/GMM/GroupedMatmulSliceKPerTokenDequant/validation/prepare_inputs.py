import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    m = param.get('m')
    k = param.get('k')
    n = param.get('n')
    groupCount = param.get('groupCount')
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    a = torch.randint(-16, 17, (m, k), device=device, dtype=torch.int8)
    b = torch.randint(-16, 17, (k, n), device=device, dtype=torch.int8)
    scale = torch.rand([groupCount, n], device=device, dtype=dtype)
    per_token_scale = torch.rand([groupCount, m], device=device, dtype=dtype)
    group_list = torch.randint(0, k + 1, (groupCount,), dtype=torch.int64, device=device)

    group_list, _ = torch.sort(group_list)

    return (a, b, scale, per_token_scale, group_list)


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

    compute_num = param.get('k')
    COMPUTE_NUM_THRESHOLD = 2048

    outputs = outputs.to(torch.float32)
    outputs_new = outputs_new.to(torch.float32)

    # Implementation note.
    rtol = RTOL_GENERAL if compute_num < COMPUTE_NUM_THRESHOLD else RTOL_OVER_THRESHOLD

    # Implementation note.
    abs_diff = torch.abs(outputs - outputs_new)

    # Implementation note.
    tolerance = rtol * torch.maximum(torch.tensor(1.0, device=outputs.device), torch.abs(outputs))

    # Implementation note.
    error_mask = abs_diff > tolerance

    # Implementation note.
    is_pass = 1 if not torch.any(error_mask) else 0

    # Implementation note.
    relative_diff = abs_diff / (torch.abs(outputs) + 1e-7)

    return is_pass, abs_diff, relative_diff