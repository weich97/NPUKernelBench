import torch
from typing import List, Tuple

def get_inputs(param, device=None) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    根据 DataFrame 行中的参数生成 ForeachLerpScalar 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状列表和数据类型。
                      Expected keys: 'input_shapes_str' (e.g., '[[10, 20], [5, 5]]'),
                      'weight_scalar', 'dtype'.
        device (torch.device): 输入张量所在设备。

    Returns:
        tuple: 包含输入张量列表 (x1_list, x2_list, weight_scalar_tensor)。
    """

    shape_list = eval(param.get('input_shape', '[[1]]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    weight_scalar = param.get('weight_scalar', 0.5)
    
    # Create lists of tensors
    x1_list = []
    x2_list = []
    for shape in shape_list:
        x1_list.append(torch.rand(shape, device=device, dtype=dtype))
        x2_list.append(torch.rand(shape, device=device, dtype=dtype))

    return (x1_list, x2_list, weight_scalar)


def get_init_inputs(param, device=None) -> List:
    """
    ForeachLerpScalar 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置。

    Returns:
        list: 空列表。
    """
    return []