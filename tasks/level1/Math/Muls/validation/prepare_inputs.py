import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    if dtype_str == "complex64":
        # 生成随机的实部和虚部（必须是 float32/float64）
        real = torch.randn(shape, dtype=torch.float32, device=device)  # 实部
        imag = torch.randn(shape, dtype=torch.float32, device=device)  # 虚部

        # 组合成复数矩阵
        x = torch.complex(real, imag)  # dtype=torch.complex64

    elif dtype == torch.int32 or dtype == torch.int64 or dtype == torch.int16:
        # 整数类型使用randint
        x = torch.randint(-100, 100, shape, device=device, dtype=dtype)
    else:
        # 浮点类型使用randn
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
    
    # 根据数据类型设置 RTOL_GENERAL
    if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
        RTOL_GENERAL = 0  # 整数类型要求精确匹配
    elif dtype == torch.float16:
        RTOL_GENERAL = 1 / 512
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 256 + 1 / 16384
    elif dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048 + 1 / 16384
    elif dtype == torch.complex64:
        # complex64 使用 float32 的容差
        RTOL_GENERAL = 1 / 2048 + 1 / 16384
    elif dtype == torch.complex128:
        # complex128 使用 float64 的容差（自定义更严格值）
        RTOL_GENERAL = 1e-12
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1

    for out, out_new in zip(outputs, outputs_new):
        # 处理复数类型
        if out.is_complex():
            # 分解实部和虚部
            out_real, out_imag = out.real, out.imag
            out_new_real, out_new_imag = out_new.real, out_new.imag
            
            # 计算实部绝对差值和相对误差
            abs_diff_real = torch.abs(out_real - out_new_real)
            rel_diff_real = abs_diff_real / (torch.abs(out_real) + 1e-7)
            
            # 计算虚部绝对差值和相对误差
            abs_diff_imag = torch.abs(out_imag - out_new_imag)
            rel_diff_imag = abs_diff_imag / (torch.abs(out_imag) + 1e-7)
            
            # 合并结果
            abs_diff = torch.cat([abs_diff_real.view(-1), abs_diff_imag.view(-1)])
            rel_diff = torch.cat([rel_diff_real.view(-1), rel_diff_imag.view(-1)])
            
            # 检查容差：分别验证实部和虚部
            tolerance_real = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out_real))
            tolerance_imag = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out_imag))
            
            if torch.any(abs_diff_real > tolerance_real) or torch.any(abs_diff_imag > tolerance_imag):
                is_pass = 0
                
        # 处理非复数类型（整数/浮点数）
        else:
            abs_diff = torch.abs(out - out_new)
            rel_diff = abs_diff / (torch.abs(out) + 1e-7)
            
            # 整数类型：检查绝对差值是否为0
            if dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
                if torch.any(abs_diff > 0):
                    is_pass = 0
            # 浮点数类型：使用相对容差
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