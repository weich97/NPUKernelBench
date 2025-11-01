from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                x1: List[torch.Tensor],
                x2: List[torch.Tensor],
                alpha: torch.Tensor) -> List[torch.Tensor]:
        """
        Foreach: out[i] = x1[i] + x2[i] * alpha
        """
        out = []
        for a, b in zip(x1, x2):   # 这里 zip 仅走列表维度，不逐元素
            if b.dtype == torch.bfloat16:
                res = (a.to(torch.float32) + b.to(torch.float32) * alpha).to(torch.bfloat16)
            else:
                res = a + b * alpha
            out.append(res)
        return out


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], alpha: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_add_list(x1,x2,alpha)

