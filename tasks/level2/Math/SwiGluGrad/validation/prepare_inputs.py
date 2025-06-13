import torch


def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 swi_glu_grad 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状和数据类型
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 包含 y_grad 和 x 两个输入张量 (y_grad, x)
    """
    shape = eval(param.get('input_shape', '[1, 2]'))  # shape 的最后一维必须为偶数
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    x = torch.rand(shape, device=device, dtype=dtype)
    y_grad = torch.rand([*shape[:-1], shape[-1] // 2], device=device, dtype=dtype)
    return (y_grad, x)


def get_init_inputs(param, device=None):
    """
    swi_glu_grad 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置

    Returns:
        list: 空列表
    """
    return []
