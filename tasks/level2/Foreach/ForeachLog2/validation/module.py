from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    Simple model that performs log2 operation.
    """

    def __init__(self):
        """
        Initialize the model for log2 operation.
        No parameters needed for basic log2 operation.
        """
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the log2 of input elements.

        Args:
            inputs: Input tensor list of any shape.

        Returns:
            Output tensor of same shape as input with log2 applied elementwise.
        """
        return [torch.log2(x) for x in inputs]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_log2(tuple(inputs))