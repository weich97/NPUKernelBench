import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    feeds_shape = eval(param.get('feeds_shape', '[1]'))
    feeds_repeat_times_shape = eval(param.get('feeds_repeat_times_shape', '[1]'))
    feeds_dtype_str = param.get('feeds_dtype', 'float16')
    feeds_repeat_times_dtype_str = param.get('feeds_repeat_times_dtype', 'int32')
    feeds_dtype = getattr(torch, feeds_dtype_str)
    feeds_repeat_times_dtype = getattr(torch, feeds_repeat_times_dtype_str)
    output_feeds_size = int(param.get('output_feeds_size', 1))

    feeds = torch.randn(feeds_shape, device=device, dtype=feeds_dtype)
    feeds_repeat_times = torch.randint(0, 10, feeds_repeat_times_shape, device=device, dtype=feeds_repeat_times_dtype)

    return (feeds, feeds_repeat_times, output_feeds_size)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for FeedsRepeat.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed