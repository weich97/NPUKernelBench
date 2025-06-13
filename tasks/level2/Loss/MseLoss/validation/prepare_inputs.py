import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # 创建预测值和目标值两个张量
    predict = torch.rand(shape, device=device, dtype=dtype)
    label = torch.rand(shape, device=device, dtype=dtype)

    return [predict, label]


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Initialization parameters for the MSE Loss model
    """
    # 从参数中获取reduction值
    reduction = param.get('reduction', 'mean')

    return [reduction]