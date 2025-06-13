from typing import List, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, reduction: str):
        super(Model, self).__init__()
        self.reduction = reduction

    def forward(self, input_predict: torch.Tensor, input_label: torch.Tensor, input_dout: torch.Tensor) -> torch.Tensor:
        # Based on mse_loss_grad_v2 logic
        if self.reduction == 'mean':
            reduce_elts = 1.0
            for dim_size in input_predict.shape:
                reduce_elts *= dim_size
            cof = (reduce_elts**(-1)) * 2.0
        elif self.reduction == 'sum': # As per standard MSE Loss, sum reduction implies cofactor 2.0
            cof = 2.0
        else: # 'none' or other reduction types
            # For 'none' reduction, MSE loss is (input_predict - input_label)^2.
            # Its gradient is 2 * (input_predict - input_label).
            cof = 2.0
        
        sub_res = input_predict - input_label
        norm_grad = sub_res * cof
        golden = norm_grad * input_dout
        return golden


class ModelNew(nn.Module):
    def __init__(self, reduction: str):
        super(ModelNew, self).__init__()
        self.reduction = reduction

    def forward(self, input_predict: torch.Tensor, input_label: torch.Tensor, input_dout: torch.Tensor) -> torch.Tensor:
        # Pass input_dout as gradOutput, input_predict as self, input_label as target
        import kernel_gen_ops
        return kernel_gen_ops.mse_loss_grad(input_dout, input_predict, input_label, self.reduction)