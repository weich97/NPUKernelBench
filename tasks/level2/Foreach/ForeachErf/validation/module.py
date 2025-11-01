from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops # Import kernel_gen_ops explicitly


class Model(nn.Module):
    """
    Simple model that performs Erf operation.
    """

    def __init__(self):
        """
        Initialize the model for Erf operation.
        No parameters needed for basic Erf operation.
        """
        super(Model, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Computes the Erf (error function) of input elements.

        Args:
            inputs: Input tensor list of any shape.

        Returns:
            Output tensor list of same shape as input with Erf applied elementwise.
        """
        # torch.erf performs the error function calculation.
        return [torch.erf(x) for x in inputs]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        # Call the NPU kernel for foreach_erf.
        # Ensure that `inputs` is passed as a tuple if `pybind.cpp` expects `c10::ArrayRef<at::Tensor>`
        # which maps well to a Python tuple of tensors or a list of tensors.
        # The common practice is to pass `tuple(inputs)` for foreach ops in `kernel_gen_ops`.
        #零开销地把 Python 端的 Tensor 列表映射到 C++ 端的 c10::ArrayRef<at::Tensor>?
        return kernel_gen_ops.foreach_erf(tuple(inputs))

