import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the model based on parameters from DataFrame row.
    """
    varRefShape = eval(param.get('varRefShape', '[1]'))
    indiceShape = eval(param.get('indiceShape', '[1]'))
    updatesShape = eval(param.get('updatesShape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    idx_dtype = torch.int64

    varRef = [torch.randn(x, device=device, dtype=dtype) for x in varRefShape]
    axis = int(param.get('axis', -2))
    indice = torch.zeros(indiceShape, device=device, dtype=idx_dtype)
    for i in range(updatesShape[0]):
        length = torch.randint(1, updatesShape[axis] + 1, (1,)).item()
        max_start_index = updatesShape[axis] - length
        start_index = torch.randint(0, max_start_index + 1, (1,)).item()
        indice[i][0], indice[i][1] = start_index, length
    updates = torch.randn(updatesShape, device=device, dtype=dtype)
    mask = torch.tensor(eval(param.get('mask', '[False]')), device=device, dtype=torch.bool)
    reduce = param.get('reduce', 'update')

    return (varRef, indice, updates, mask, reduce, axis)

def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for ScatterList.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed