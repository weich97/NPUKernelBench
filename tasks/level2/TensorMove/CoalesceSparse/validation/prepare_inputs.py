import torch
import torch_npu
import numpy as np

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 CoalesceSparse 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状和数据类型
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 张量 (indices, values, sparse_tensor)
    """

    indices_shape = eval(param.get('indices_shape', '[2, 2]'))
    indices_dtype_str = param.get('indices_dtype', 'int32')
    indices_dtype = getattr(torch, indices_dtype_str)
    indices = torch.randint(0, 10, indices_shape, device=device, dtype=indices_dtype)

    values_shape = eval(param.get('values_shape', '[2, 2]'))
    values_dtype_str = param.get('values_dtype', 'float16')
    values_dtype = getattr(torch, values_dtype_str)
    values = torch.randn(values_shape, device=device, dtype=values_dtype)

    max_values, _ = torch.max(indices, dim=1)
    max_indices = (max_values + 1).tolist()
    sparse_dim = indices.shape[0]
    dense_dim = values.dim() - 1
    start_idx = len(max_indices)
    for i in range(start_idx, sparse_dim + dense_dim):
        max_indices.append(values.shape[i - start_idx + 1])

    sparse_tensor = torch.sparse_coo_tensor(indices, values, max_indices)

    return (indices, values, sparse_tensor)


def get_init_inputs(param, device=None):
    """
    CoalesceSparse 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置

    Returns:
        list: 空列表
    """
    return []