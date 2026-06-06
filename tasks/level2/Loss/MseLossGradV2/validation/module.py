from typing import List, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, reduction: str):
        super(Model, self).__init__()
        self.reduction = reduction

    def forward(
        self,
        input_predict: torch.Tensor,
        input_label:   torch.Tensor,
        input_dout:    torch.Tensor
    ) -> torch.Tensor:
        # Implementation note.
        pred_f32 = input_predict.to(torch.float32)
        tgt_f32  = input_label.to(torch.float32)
        dout_f32 = input_dout.to(torch.float32)

        # Implementation note.
        if self.reduction == 'mean':
            cof = torch.scalar_tensor(2.0 / pred_f32.numel(),
                                    device=pred_f32.device,
                                    dtype=torch.float32)
        elif self.reduction == 'sum':
            cof = torch.scalar_tensor(2.0,
                                    device=pred_f32.device,
                                    dtype=torch.float32)
        else:  # 'none'
            # Implementation note.
            cof = torch.tensor(2.0, device=pred_f32.device, dtype=torch.float32)

        # Implementation note.
        sub_res   = pred_f32 - tgt_f32  # Implementation note.
        norm_grad = sub_res * cof  # Implementation note.
        golden    = norm_grad * dout_f32  # Implementation note.

        # Implementation note.
        return golden.to(input_predict.dtype)


class ModelNew(nn.Module):
    def __init__(self, reduction: str):
        super(ModelNew, self).__init__()
        self.reduction = reduction

    def forward(self, input_predict: torch.Tensor, input_label: torch.Tensor, input_dout: torch.Tensor) -> torch.Tensor:
        # Pass input_dout as gradOutput, input_predict as self, input_label as target
        import kernel_gen_ops
        return kernel_gen_ops.mse_loss_grad(input_dout, input_predict, input_label, self.reduction)