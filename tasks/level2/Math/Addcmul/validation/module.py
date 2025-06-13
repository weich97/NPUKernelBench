import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input_data: torch.Tensor, input_x1: torch.Tensor, input_x2: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        output = (input_data + input_x1 * input_x2 * value)
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input_data: torch.Tensor, input_x1: torch.Tensor, input_x2: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.addcmul(input_data, input_x1, input_x2, value)
