import torch

from framework.utils import check_precision


def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 SwiGlu 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状和数据类型
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 包含一个输入张量 (x,)
    """
    shape = eval(param.get('input_shape', '[1, 2]'))  # 默认至少是chunkable的shape
    dtype_str = param.get('dtype', 'float16')
    dim = param.get('dim', -1)
    
    dtype = getattr(torch, dtype_str)
    x = torch.rand(shape, device=device, dtype=dtype)
    return (x, dim)


def get_init_inputs(param, device=None):
    """
    SwiGlu 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置

    Returns:
        list: 空列表
    """
    return []


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float32:
        return check_precision(outputs, outputs_new, max_abs_error=0.00001, max_rel_error=0.00001)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.01, max_rel_error=0.01)
