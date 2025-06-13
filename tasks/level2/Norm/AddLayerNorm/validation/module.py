from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn



class Model(nn.Module):

    def __init__(self, gamma: torch.Tensor, beta: torch.Tensor, epsilon: float, additional_out: bool):
        super(Model, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(
            gamma.shape,
            eps=epsilon,
            elementwise_affine=True
        )
        self.layer_norm.weight.data = gamma
        self.layer_norm.bias.data = beta
        self.additional_out = additional_out

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, bias: torch.Tensor) -> List[torch.Tensor]:
        x = x1 + x2
        if bias is not None:
            x = x + bias
        if self.additional_out:
            return [x, self.layer_norm(x)]
        else:
            return [self.layer_norm(x)]


class ModelNew(nn.Module):
    def __init__(self, gamma, beta, epsilon: float, additional_out: bool):
        super(ModelNew, self).__init__()
        self.gamma = gamma
        self.beta = beta
        self.epsilon = epsilon
        self.additional_out = additional_out

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, bias: torch.Tensor) -> List[torch.Tensor]:
        import kernel_gen_ops
        return kernel_gen_ops.add_layer_norm(x1, x2, bias, self.gamma, self.beta, self.epsilon, self.additional_out)
