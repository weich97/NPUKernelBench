import torch
import numpy as np


def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 ReverseSequence 算子的输入张量。

    Args:
        param (dict): 参数配置，如输入形状、序列长度、维度等
        device (torch.device): 输入张量所在设备

    Returns:
        tuple: 包含输入张量 (x, seq_lengths, seq_dim, batch_dim)
    """
    shape = eval(param.get('input_shape', '[3, 5, 7]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    seq_dim = param.get('seq_dim', 1)  # 直接获取，默认值为整数 1
    batch_dim = param.get('batch_dim', 0) # 直接获取，默认值为整数 0

    max_len = shape[seq_dim]
    batch_size = shape[batch_dim]

    # 生成随机的序列长度，但不能超过 seq_dim 的长度
    seq_lengths_np = np.random.randint(1, max_len + 1, size=batch_size)
    seq_lengths = torch.tensor(seq_lengths_np, dtype=torch.int64, device=device)

    x = torch.randn(shape, device=device, dtype=dtype)
    return (x, seq_lengths, seq_dim, batch_dim)


def get_init_inputs(param, device=None):
    """
    reverse_sequence 没有模型初始化参数，返回空列表。

    Args:
        param (dict): 参数配置

    Returns:
        list: 空列表
    """
    return []