from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()

    def forward(self, feeds: torch.Tensor, feeds_repeat_times: torch.Tensor, output_feeds_size: int) -> torch.Tensor:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """

        # Implementation note.
        repeated = torch.repeat_interleave(feeds, feeds_repeat_times, dim=0)

        # Implementation note.
        total_repeated = feeds_repeat_times.sum().item()
        pad_size = output_feeds_size - total_repeated

        # Implementation note.
        if pad_size > 0:
            # Implementation note.
            output_shape = (output_feeds_size,) + feeds.shape[1:]
            output = torch.zeros(output_shape,
                                 dtype=feeds.dtype,
                                 device=feeds.device)

            # Implementation note.
            output[:total_repeated] = repeated
            return output
        else:
            # Implementation note.
            return repeated


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, feeds: torch.Tensor, feeds_repeat_times: torch.Tensor, output_feeds_size: int) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.feeds_repeat(feeds, feeds_repeat_times, output_feeds_size)
