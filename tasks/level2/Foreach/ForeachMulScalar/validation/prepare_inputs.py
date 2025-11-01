import torch


def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成模型的输入张量列表和标量。
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    scalar_value = float(param.get('scalar', '1.0'))  # 使用单个标量
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    inputs = []
    for shape in shape_list:
        if dtype == torch.int32:
            # 整数类型使用 randint
            x = torch.randint(-100, 100, shape, device=device, dtype=dtype)
        else:
            # 浮点类型使用 randn
            x = torch.randn(shape, device=device, dtype=dtype)
        inputs.append(x)

    # 根据文档规则设置标量的数据类型：
    # - 当x的数据类型为FLOAT、FLOAT16、INT32时，数据类型与x的数据类型保持一致
    # - 当x的数据类型为BFLOAT16时，数据类型支持FLOAT
    if dtype == torch.bfloat16:
        scalar = torch.tensor(scalar_value, device=device, dtype=torch.float)
    elif dtype == torch.int32:
        scalar = torch.tensor(int(scalar_value), device=device, dtype=torch.int32)
    else:  # float或float16
        scalar = torch.tensor(scalar_value, device=device, dtype=dtype)

    return inputs, scalar


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for foreach_mul_scalar.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed