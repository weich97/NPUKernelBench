from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()

    def forward(
        self,
        x1: List[torch.Tensor],
        x2: List[torch.Tensor],
        x3: List[torch.Tensor],
        scalars: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Reference implementation detail.

        Reference implementation detail.
        Reference implementation detail.
        Reference implementation detail.
        Reference implementation detail.
        Reference implementation detail.
        """
        ret = []
        for scalar, x, y, z in zip(scalars, x1, x2, x3):
            if x.dtype == torch.bfloat16:
                ret.append(x.to(torch.float32) + scalar.to(torch.float32) * y.to(torch.float32) * z.to(torch.float32))
            else:
                ret.append(x + scalar * y * z)
        return ret

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalars: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_addcmul_list(x1, x2, x3, scalars)