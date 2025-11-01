from typing import List
import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grads: List[torch.Tensor], exponent: torch.Tensor) -> List[torch.Tensor]:
        return [x ** exponent.item() for x in grads]


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, grads: List[torch.Tensor], exponent: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_pow_scalar(grads, exponent)