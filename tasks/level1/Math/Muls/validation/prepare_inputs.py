import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    if dtype_str == "complex64":
        # Implementation note.
        real = torch.randn(shape, dtype=torch.float32, device=device)  # Implementation note.
        imag = torch.randn(shape, dtype=torch.float32, device=device)  # Implementation note.

        # Implementation note.
        x = torch.complex(real, imag)  # dtype=torch.complex64

    elif dtype == torch.int32 or dtype == torch.int64 or dtype == torch.int16:
        # Implementation note.
        x = torch.randint(-100, 100, shape, device=device, dtype=dtype)
    else:
        # Implementation note.
        x = torch.randn(shape, device=device, dtype=dtype)

    value_value = float(param.get('value', 1.0))
    value = torch.tensor([value_value], device=device, dtype=float)

    return x, value


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for Muls.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    # Implementation note.
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
        RTOL_GENERAL = 0  # Implementation note.
    elif dtype == torch.float16:
        RTOL_GENERAL = 1 / 512
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 256 + 1 / 16384
    elif dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048 + 1 / 16384
    elif dtype == torch.complex64:
        # Implementation note.
        RTOL_GENERAL = 1 / 2048 + 1 / 16384
    elif dtype == torch.complex128:
        # Implementation note.
        RTOL_GENERAL = 1e-12
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1

    for out, out_new in zip(outputs, outputs_new):
        # Implementation note.
        if out.is_complex():
            # Implementation note.
            out_real, out_imag = out.real, out.imag
            out_new_real, out_new_imag = out_new.real, out_new.imag
            
            # Implementation note.
            abs_diff_real = torch.abs(out_real - out_new_real)
            rel_diff_real = abs_diff_real / (torch.abs(out_real) + 1e-7)
            
            # Implementation note.
            abs_diff_imag = torch.abs(out_imag - out_new_imag)
            rel_diff_imag = abs_diff_imag / (torch.abs(out_imag) + 1e-7)
            
            # Implementation note.
            abs_diff = torch.cat([abs_diff_real.view(-1), abs_diff_imag.view(-1)])
            rel_diff = torch.cat([rel_diff_real.view(-1), rel_diff_imag.view(-1)])
            
            # Implementation note.
            tolerance_real = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out_real))
            tolerance_imag = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out_imag))
            
            if torch.any(abs_diff_real > tolerance_real) or torch.any(abs_diff_imag > tolerance_imag):
                is_pass = 0
                
        # Implementation note.
        else:
            abs_diff = torch.abs(out - out_new)
            rel_diff = abs_diff / (torch.abs(out) + 1e-7)
            
            # Implementation note.
            if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
                if torch.any(abs_diff > 0):
                    is_pass = 0
            # Implementation note.
            else:
                tolerance = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out))
                if torch.any(abs_diff > tolerance):
                    is_pass = 0
            
            abs_diff = abs_diff.view(-1)
            rel_diff = rel_diff.view(-1)
        
        all_abs_diff.append(abs_diff)
        all_rel_diff.append(rel_diff)

    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff