import torch
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        softmax = nn.Softmax(dim=-1)
        output = softmax(input)
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.inplace_attn_softmax(input)