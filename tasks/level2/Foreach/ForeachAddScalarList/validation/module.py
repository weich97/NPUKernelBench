from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    Simple model that performs log operation.
    """

    def __init__(self):
        """
        Initialize the model for log operation.
        No parameters needed for basic log operation.
        """
        super(Model, self).__init__()

    def forward(
        self,
        x: List[torch.Tensor],
        alpha: torch.Tensor,
    ) -> List[torch.Tensor]:
        """
        逐张量计算：out_i = x_i + α_i
        α 是一维 Tensor，长度等于 len(x)。
        中间值用 float32，结果回到原 dtype。
        """
        return [
            (t.float() + alpha[i]).to(t.dtype)
            for i, t in enumerate(x)
        ]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: List[torch.Tensor], alpha: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_add_scalar_list(x, alpha)

