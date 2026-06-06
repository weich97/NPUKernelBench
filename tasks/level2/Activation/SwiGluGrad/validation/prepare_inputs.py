import torch


def get_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
    shape = eval(param.get('input_shape', '[1, 2]'))  # Implementation note.
    dtype_str = param.get('dtype', 'float16')
    dim = param.get('dim', -1)
    
    dtype = getattr(torch, dtype_str)
    x = torch.rand(shape, device=device, dtype=dtype)
    
    y_grad_shape = list(shape)
    y_grad_shape[dim] //= 2
    y_grad = torch.rand(y_grad_shape, device=device, dtype=dtype)
    
    return (y_grad, x, dim)


def get_init_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
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

    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1
    for out, out_new in zip(outputs, outputs_new):
        # Implementation note.
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        all_abs_diff.append(abs_diff.view(-1))
        all_rel_diff.append(rel_diff.view(-1))

        # Implementation note.
        tolerance = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out))

        # Implementation note.
        error_mask = abs_diff > tolerance

        # Implementation note.
        if torch.any(error_mask):
            is_pass = 0


    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)

    return is_pass, all_abs_diff, all_rel_diff