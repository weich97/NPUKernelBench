import torch
import torch.nn as nn
import kernel_gen_ops
from typing import List

class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, xs: list[torch.Tensor]) -> torch.Tensor:
        # 对每个张量单独判断是否有 NaN/Inf
        flags = [float(torch.isnan(x).any() or torch.isinf(x).any()) for x in xs]
        return torch.tensor(sum(flags) > 0, dtype=torch.float32)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: List[torch.Tensor]) -> torch.Tensor:
        return kernel_gen_ops.non_finite_check_op(x)