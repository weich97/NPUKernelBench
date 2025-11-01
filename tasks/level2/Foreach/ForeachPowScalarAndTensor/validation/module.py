from typing import List, Tuple
import torch
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: List[torch.Tensor], scalar_weight) -> List[torch.Tensor]:
        """
        """
        if not isinstance(x, list):
            raise TypeError("Inputs x must be lists of tensors.")

        output_list = []
        for i in range(len(x)):
            result_tensor = torch.pow(scalar_weight, x[i])
            output_list.append(result_tensor)
        return output_list


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()
        
    def forward(self, x: List[torch.Tensor], scalar_weight) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_pow_scalar_and_tensor(x, torch.tensor(scalar_weight))