import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    if dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        # 根据类型确定合适的范围
        if dtype == torch.int8:
            low, high = -128, 127
        elif dtype == torch.uint8:
            low, high = 0, 255
        else:
            low, high = -100, 100
        x = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype == torch.bool:
        x = torch.randint(0, 2, shape, device=device, dtype=torch.bool)
    else:
        # 浮点类型使用randn
        x = torch.randn(shape, device=device, dtype=dtype)

    dst_type_str = param.get('dst_type', 'float16')
    dst_dtype = getattr(torch, dst_type_str)
    type_map = {
        torch.float32: 0,
        torch.float16: 1,
        torch.int8: 2,
        torch.int32: 3,
        torch.uint8: 4,
        torch.int16: 6,
        torch.int64: 9,
        torch.bool: 12,
        torch.bfloat16: 27
    }
    return [x, type_map.get(dst_dtype, torch.float16)]


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for Cast.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed