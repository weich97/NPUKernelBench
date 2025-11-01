from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn



class Model(nn.Module):
    """
    Simple model that performs copy operation.
    """

    def __init__(self):
        """
        Initialize the model for copy operation.
        No parameters needed for basic copy operation.
        """
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the copy of input elements.

        Args:
            inputs: Input tensor list of any shape.

        Returns:
            It's simply creating a copy of each input tensor.
        """
        return [torch.clone(x) for x in inputs]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        import kernel_gen_ops
        return kernel_gen_ops.foreach_copy(tuple(inputs))

