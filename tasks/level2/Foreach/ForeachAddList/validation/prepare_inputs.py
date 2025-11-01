import torch


def get_inputs(param, device=None):
    shape_list = eval(param.get('input_shape', '[[1]]'))
    scalar = param.get('scalars', 1.0)
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    compute_dtype = torch.float32 if dtype == torch.bfloat16 else dtype

    x1 = []
    x2 = []
    for shape in shape_list:
        x1.append(torch.randn(shape, device=device, dtype=dtype))
        x2.append(torch.randn(shape, device=device, dtype=dtype))

    # 把 scalars_list 变成 1-D Tensor，长度=列表数
    alpha = torch.tensor(scalar, device=device, dtype=compute_dtype)

    return x1, x2, alpha 


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

