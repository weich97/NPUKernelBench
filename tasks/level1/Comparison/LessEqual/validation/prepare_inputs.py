import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    shape = eval(param.get('shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    
    input1 = torch.rand(shape, device=device, dtype=dtype)

    similarity = torch.empty(1).uniform_(0, 1).item() #控制input1 和 input2的相似度

    total_num = torch.tensor(shape).prod().item()
    num_same = int(total_num * similarity)
    num_diff = total_num - num_same

    input2 = input1.clone()
    
    indices = torch.randperm(total_num, device=device)[:num_diff]
    input2_flat = input2.view(-1)
    random_values = torch.randn(num_diff, dtype=dtype, device=device)
    input2_flat[indices] = random_values
    input2 = input2_flat.view(shape)

    return (input1, input2)


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
