import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # 创建随机张量列表
    x1 = []
    x2 = []
    x3 = []
    for shape in shape_list:
        if dtype == torch.int32:
            # 整数类型使用randint
            x = torch.randint(-100, 100, shape, device=device, dtype=dtype)
            y = torch.randint(-100, 100, shape, device=device, dtype=dtype)
            z = torch.randint(-100, 100, shape, device=device, dtype=dtype)
        else:
            # 浮点类型使用randn
            x = torch.rand(shape, device=device, dtype=dtype)
            y = torch.rand(shape, device=device, dtype=dtype)
            z = torch.rand(shape, device=device, dtype=dtype)

        x1.append(x)
        x2.append(y)
        x3.append(z)

    scalars_value = eval(param.get('scalars', '[1.0]'))
    scalars = torch.tensor(scalars_value, device=device, dtype=dtype)

    return x1, x2, x3, scalars


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

from framework.utils import check_precision

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