import torch

def get_inputs(param, device=None):
    """
    返回：
        x      : List[Tensor] 与 input_shape 一一对应
        alpha  : 1-D Tensor    长度=len(shape_list)，dtype按映射表
    """
    shape_list = eval(param.get('input_shape', '[[1]]'))
    scale = eval(param.get('scale', '[1.0]'))
    dtype_str = param.get('dtype', 'float16')
    x_dtype = getattr(torch, dtype_str)

    # 编译器规定的 alpha dtype 映射
    alpha_dtype_map = {
        torch.float16: torch.float32,
        torch.float32: torch.float32,
        torch.int32:   torch.int64,
        torch.int64:   torch.int64,
        torch.bfloat16: torch.float32,
    }
    alpha_dtype = alpha_dtype_map.get(x_dtype, torch.float32)

    # 生成 x（保持原 dtype）
    x = [torch.rand(s, device=device, dtype=x_dtype) for s in shape_list]
    alpha = torch.tensor(scale, dtype=alpha_dtype)
    return x, alpha


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