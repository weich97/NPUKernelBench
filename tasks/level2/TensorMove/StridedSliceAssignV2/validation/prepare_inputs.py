import torch


def get_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
    var_ref_shape = eval(param.get('var_ref_shape', '[10, 10]'))
    input_value_shape = eval(param.get('input_value_shape', '[5, 10]'))
    begin_val = eval(param.get('begin_val', '[0]'))
    end_val = eval(param.get('end_val', '[5]'))
    strides_val = eval(param.get('strides_val', '[1]'))
    axes_optional_val = eval(param.get('axes_optional_val', '[0]'))

    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    var_ref = torch.rand(var_ref_shape, device=device, dtype=dtype)
    input_value = torch.rand(input_value_shape, device=device, dtype=dtype)
    # Implementation note.
    begin = torch.tensor(begin_val, device=device, dtype=torch.int64)
    end = torch.tensor(end_val, device=device, dtype=torch.int64)
    strides = torch.tensor(strides_val, device=device, dtype=torch.int64)
    axes_optional = torch.tensor(axes_optional_val, device=device, dtype=torch.int64)

    return (var_ref, input_value, begin, end, strides, axes_optional)


def get_init_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
    return []