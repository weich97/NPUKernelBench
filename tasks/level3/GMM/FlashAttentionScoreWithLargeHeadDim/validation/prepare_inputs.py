import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    head_num = param.get('head_num')
    scale_value = param.get('scale_value', 1.0)

    # 创建query、key和value张量，所有元素的值在-0.01~0.01之间随机，数据类型为float16
    query = torch.rand(shape, device=device, dtype=torch.float16) * 0.02 - 0.01
    key = torch.rand(shape, device=device, dtype=torch.float16) * 0.02 - 0.01
    value = torch.rand(shape, device=device, dtype=torch.float16) * 0.02 - 0.01

    return (query, key, value, scale_value, head_num)


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
