import torch
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, input: torch.Tensor, clip_value_min: torch.Tensor, clip_value_max: torch.Tensor) -> torch.Tensor:

        # input = input.cpu().numpy()
        # clip_value_min = clip_value_min.cpu().numpy()
        # clip_value_max = clip_value_max.cpu().numpy()
        # output = tf.clip_by_value(input, clip_value_min, clip_value_max)
        output = torch.clamp(input, min=clip_value_min, max=clip_value_max)
        return output

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, input: torch.Tensor, clip_value_min: torch.Tensor, clip_value_max: torch.Tensor) -> torch.Tensor:
        return kernel_gen_ops.clip_by_value(input, clip_value_min, clip_value_max)