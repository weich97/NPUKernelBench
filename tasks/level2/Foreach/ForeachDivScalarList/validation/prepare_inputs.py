import torch



def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成模型的输入张量和标量列表。
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    scalar_list_values = eval(param.get('scalar_list', '[1.0]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    inputs = []
    for shape in shape_list:

        x = torch.randn(shape, device=device, dtype=dtype)
        inputs.append(x)

    scalar_list = torch.tensor(scalar_list_values, device="cpu:0", dtype=torch.float)

    return inputs, scalar_list


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