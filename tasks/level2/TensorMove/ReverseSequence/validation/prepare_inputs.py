import torch
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
    shape = eval(param.get('input_shape', '[3, 5, 7]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    seq_dim = param.get('seq_dim', 1)  # Implementation note.
    batch_dim = param.get('batch_dim', 0)  # Implementation note.

    max_len = shape[seq_dim]
    batch_size = shape[batch_dim]

    # Implementation note.
    seq_lengths_np = np.random.randint(1, max_len + 1, size=batch_size)
    seq_lengths = torch.tensor(seq_lengths_np, dtype=torch.int64, device=device)

    x = torch.randn(shape, device=device, dtype=dtype)
    return (x, seq_lengths, seq_dim, batch_dim)


def get_init_inputs(param, device=None):
    """
    Reference implementation detail.

    Args:
        Reference implementation detail.

    Returns:
        Reference implementation detail.
    """
    return []