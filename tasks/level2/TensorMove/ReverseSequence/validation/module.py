from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import torch.nn.functional as F

import kernel_gen_ops

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, x, seq_lengths, seq_dim=1, batch_dim=0):
        input_shape = x.shape
        output = torch.zeros_like(x)
        batch_size = input_shape[batch_dim]

        for i in range(batch_size):
            batch_selector = [slice(None)] * len(input_shape)
            batch_selector[batch_dim] = i
            batch_selector = tuple(batch_selector)

            seq_len = seq_lengths[i].item() if seq_lengths.ndim > 0 else seq_lengths

            reversed_indices = torch.arange(seq_len - 1, -1, -1, device=x.device)

            # Create indices for the sequence dimension
            seq_indices = torch.arange(seq_len, device=x.device)

            selector = list(batch_selector)
            selector[seq_dim] = seq_indices
            selector = tuple(selector)

            reversed_selector = list(batch_selector)
            reversed_selector[seq_dim] = reversed_indices
            reversed_selector = tuple(reversed_selector)

            output[selector] = x[reversed_selector]

            if seq_len < input_shape[seq_dim]:
                remaining_selector = list(batch_selector)
                remaining_selector[seq_dim] = slice(seq_len, None)
                remaining_selector = tuple(remaining_selector)
                output[remaining_selector] = x[remaining_selector]

        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: torch.Tensor, seq_lengths: torch.Tensor, seq_dim: int = 1, batch_dim: int = 0) -> torch.Tensor:
        return kernel_gen_ops.reverse_sequence(x, seq_lengths, seq_dim, batch_dim)