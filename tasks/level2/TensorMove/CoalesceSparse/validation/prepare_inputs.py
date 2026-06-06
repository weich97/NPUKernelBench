import torch
import torch_npu
import numpy as np

def get_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """

    indices_shape = eval(param.get('indices_shape', '[2, 2]'))
    indices_dtype_str = param.get('indices_dtype', 'int32')
    indices_dtype = getattr(torch, indices_dtype_str)
    indices = torch.randint(0, 10, indices_shape, device=device, dtype=indices_dtype)

    values_shape = eval(param.get('values_shape', '[2, 2]'))
    values_dtype_str = param.get('values_dtype', 'float16')
    values_dtype = getattr(torch, values_dtype_str)
    values = torch.randn(values_shape, device=device, dtype=values_dtype)

    max_values, _ = torch.max(indices, dim=1)
    max_indices = (max_values + 1).tolist()
    sparse_dim = indices.shape[0]
    dense_dim = values.dim() - 1
    start_idx = len(max_indices)
    for i in range(start_idx, sparse_dim + dense_dim):
        max_indices.append(values.shape[i - start_idx + 1])

    sparse_tensor = torch.sparse_coo_tensor(indices, values, max_indices)

    return (indices, values, sparse_tensor)


def get_init_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
    return []