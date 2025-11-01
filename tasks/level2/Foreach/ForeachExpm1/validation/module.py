from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    Simple model that performs expm1 operation.
    """

    def __init__(self):
        """
        Initialize the model for expm1 operation.
        No parameters needed for basic expm1 operation.
        """
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the expm1 of input elements.

        Args:
            inputs: Input tensor list of any shape.

        Returns:
            Output tensor of same shape as input with expm1 applied elementwise.
        """
        return [torch.expm1(x) for x in inputs]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_expm1(tuple(inputs))
