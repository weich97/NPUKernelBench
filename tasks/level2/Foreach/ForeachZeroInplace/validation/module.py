from typing import List
import numpy as np # Import numpy for np.zeros_like

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    Simple model that creates new tensors filled with zeros, matching the shape and dtype of the input tensors.
    """

    def __init__(self):
        """
        Initializes the model for the zeroing operation.
        No parameters are needed for this basic operation.
        """
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Creates new tensors filled with zeros, matching the shape and dtype of the input tensors.
        This operation is non-inplace relative to the original input tensors.

        Args:
            inputs: A list of input tensors.

        Returns:
            A new list of tensors, each filled with zeros, copying the shape and dtype of the corresponding input.
        """
        result_list = []
        for x in inputs:
            # Create a new tensor filled with zeros, matching the shape and dtype of the input tensor
            zeroed_tensor = torch.zeros_like(x)
            result_list.append(zeroed_tensor)
        return result_list


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_zero_inplace(tuple(inputs))

