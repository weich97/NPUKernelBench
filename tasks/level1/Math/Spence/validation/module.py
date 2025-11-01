import torch
import torch.nn as nn

from scipy.special import spence

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        device = input.device
        input = input.cpu().numpy()
        output = spence(input)
        output = torch.from_numpy(output)
        return output.to(device)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.spence(input)