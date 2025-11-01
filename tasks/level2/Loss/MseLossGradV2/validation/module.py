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
        # 1. 统一提升到 float32
        pred_f32 = input_predict.to(torch.float32)
        tgt_f32  = input_label.to(torch.float32)
        dout_f32 = input_dout.to(torch.float32)

        # 2. 计算梯度系数 cof
        if self.reduction == 'mean':
            cof = torch.scalar_tensor(2.0 / pred_f32.numel(),
                                    device=pred_f32.device,
                                    dtype=torch.float32)
        elif self.reduction == 'sum':
            cof = torch.scalar_tensor(2.0,
                                    device=pred_f32.device,
                                    dtype=torch.float32)
        else:  # 'none'
            # 逐元素系数 2.0，自动广播到与 pred_f32 同形
            cof = torch.tensor(2.0, device=pred_f32.device, dtype=torch.float32)

        # 3. 计算梯度
        sub_res   = pred_f32 - tgt_f32          # (...,) 与输入同形
        norm_grad = sub_res * cof               # 广播后同形
        golden    = norm_grad * dout_f32        # 与 input_dout 广播规则一致

        # 4. 返回原 dtype
        return golden.to(input_predict.dtype)


class ModelNew(nn.Module):
    def __init__(self, reduction: str):
        super(ModelNew, self).__init__()
        self.reduction = reduction

    def forward(self, input_predict: torch.Tensor, input_label: torch.Tensor, input_dout: torch.Tensor) -> torch.Tensor:
        # Pass input_dout as gradOutput, input_predict as self, input_label as target
        import kernel_gen_ops
        return kernel_gen_ops.mse_loss_grad(input_dout, input_predict, input_label, self.reduction)