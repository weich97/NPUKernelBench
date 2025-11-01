import torch
from typing import Optional, Tuple
import numpy as np

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 DequantSwigluQuant 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状和数据类型
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 包含 DequantSwigluQuant 算子的所有输入张量和非张量参数
               (x, weight_scale, activate_scale, bias, quant_scale, quant_offset, group_index, activate_left, quant_mode)
    """
    # 必选参数
    # input_shape 仍然可能是字符串 "[1, 2]"，所以 eval 是正确的
    shape = eval(param.get('input_shape', '[1, 2]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    x = torch.rand(shape, device=device, dtype=dtype)

    activate_left = param.get('activate_left', True)
    quant_mode = param.get('quant_mode', 'static')

    group_index = None
    if param.get('group_index_present', False):
        group_index_val = x.shape[0] # Assuming x is already (N_flattened, H)
        count = int(param.get('group', 1))
        group_list = torch.randint(0, group_index_val + 1, (count,), dtype=torch.int32, device=device)
        group_list[-1] = group_index_val
        group_list, _ = torch.sort(group_list)
        group_list = [group_list[0]] + [group_list[i] - group_list[i-1] for i in range(1, len(group_list))]
        group_index = torch.tensor(group_list, device=device, dtype=torch.int32)
    else:
        count = 1

    weight_scale = None
    if param.get('weight_scale_present', False):
        weight_scale_shape = [count, shape[-1]]
        weight_scale = torch.rand(weight_scale_shape, device=device, dtype=torch.float32)

    activate_scale = None
    if param.get('activate_scale_present', False):
        activate_scale_shape = list(shape[:-1]) + [1]
        activate_scale = torch.rand(activate_scale_shape, device=device, dtype=torch.float32)

    bias = None
    if param.get('bias_present', False):
        bias_shape = [shape[-1]]
        bias_dtype_str = param.get('dtype', 'float16')
        bias_dtype = getattr(torch, bias_dtype_str)
        bias = torch.rand(bias_shape, device=device, dtype=bias_dtype)

    quant_scale = None
    if param.get('quant_scale_present', True):
        quant_scale_shape = [count, x.shape[-1] // 2]
        quant_scale = torch.rand(quant_scale_shape, device=device, dtype=torch.float32)

    quant_offset = None
    if param.get('quant_offset_present', True):
        if quant_mode == "static":
            quant_offset = torch.rand([count, x.shape[-1] // 2], device=device, dtype=torch.float32)


    return (x,
            weight_scale,
            activate_scale,
            bias,
            quant_scale,
            quant_offset,
            group_index,
            activate_left, # This is now correctly a bool
            quant_mode)    # This is now a str

def get_init_inputs(param, device=None):
    """
    DequantSwigluQuant 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置

    Returns:
        list: 空列表
    """
    return []