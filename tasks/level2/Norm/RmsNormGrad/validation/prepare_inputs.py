import torch

def get_inputs(param, device=None):
    input_shape = eval(param.get('input_shape', '[1]'))
    gamma_shape = eval(param.get('gamma_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    epsilon = param.get('epsilon', 1e-6)

    x = torch.rand(input_shape, device=device, dtype=dtype)
    dy = torch.rand(input_shape, device=device, dtype=dtype)
    gamma = torch.rand(gamma_shape, device=device, dtype=torch.float32)

    rstd_shape = list(input_shape)
    for i in range(len(gamma_shape)):
        rstd_shape[len(input_shape) - 1 - i] = 1

    rstd = torch.rsqrt(x.pow(2).mean(dim=tuple(range(len(input_shape) - len(gamma_shape), len(input_shape))), keepdim=True) + epsilon).to(torch.float32)
    rstd = rstd.view(rstd_shape).to(device=device, dtype=torch.float32)

    return (dy, x, rstd, gamma)

def get_init_inputs(param, device=None):
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