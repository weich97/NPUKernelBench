# prepare_inputs.py
import torch
import ast  # 更安全地处理字符串转列表
import torch.nn.functional as F

def get_inputs(param, device=None):
    """
    生成 GeGLUGradV2 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状和数据类型
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 包含输入张量 (dy, x, gelu_param_tensor, dim, approximate, activateLeft)
        # Note: gelu_param_tensor 是为了匹配 Model.forward 的 gelu_output 参数
    """
    shape_str = param.get('input_shape', '[1, 2]')
    shape = ast.literal_eval(shape_str)

    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    dim = int(param.get('dim', -1))

    # approximate 现在直接返回整数
    approximate = int(param.get('approximate', 0))

    activateLeft = bool(param.get('activateLeft', True))

    dy = torch.rand(shape, device=device, dtype=dtype)

    x_shape = list(shape)
    # 调整 x_shape 以匹配 GeGLU 的输入（最后一维通常是两倍）
    # 假设 dim 是最后一个维度，如果不是，需要更复杂的逻辑来处理中间维度
    if dim == -1:
        x_shape[-1] = x_shape[-1] * 2
    elif dim < len(x_shape):
        x_shape[dim] = x_shape[dim] * 2
    else:
        # 如果 dim 超出范围，这可能是个错误，或者需要特别处理
        raise ValueError(f"Invalid dim: {dim} for shape {shape}")

    # Create x tensor with requires_grad=True from the start
    x = torch.rand(x_shape, device=device, dtype=dtype, requires_grad=True)

    return (dy, x, dim, approximate, activateLeft)


def get_init_inputs(param, device=None):
    """
    GeGluGradV2 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置

    Returns:
        list: 空列表
    """
    return []