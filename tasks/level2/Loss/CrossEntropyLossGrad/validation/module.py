# module.py
from typing import List, Optional, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor], ignore_index: int, label_smoothing: float,
                 reduction: str, lse_square_scale_for_zloss: float):
        super(Model, self).__init__()
        # PyTorch CrossEntropyLoss expects weight to be float
        self.weight = weight 
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.lse_square_scale_for_zloss = lse_square_scale_for_zloss 

    def forward(
        self,
        grad_loss: torch.Tensor,          # [N] 或 []
        log_softmax: torch.Tensor,        # [N, C]
        target: torch.Tensor,             # [N]
        grad_zloss: Optional[torch.Tensor],   # [C]
        lse_for_zloss: Optional[torch.Tensor]
    ) -> List[torch.Tensor]:

        # ---------------- 统一转 float32 ----------------
        log_softmax_fp32 = log_softmax.to(torch.float32)
        grad_loss_fp32   = grad_loss.to(torch.float32)
        weight_fp32      = self.weight.to(torch.float32) if self.weight is not None \
                           else torch.ones(log_softmax.size(-1), dtype=torch.float32, device=log_softmax.device)
        target_fp32      = target.to(torch.int64)

        batch_size, num_classes = log_softmax_fp32.shape

        # --------------- 参考代码逻辑开始 ---------------
        # 1. 根据 target 取出对应 weight
        weight_yn = torch.gather(weight_fp32, 0, target_fp32)          # [N]

        # 2. ignore_index mask
        if self.ignore_index >= 0:
            ignore_mask = (target_fp32 != self.ignore_index).float()        # [N]
        else:
            ignore_mask = torch.ones(batch_size, dtype=torch.float32, device=log_softmax.device)

        # 3. 计算 loss_out_grad 与 smooth_loss_grad
        if self.reduction == "mean":
            mean_out_grad   = grad_loss_fp32 * (1.0 - self.label_smoothing)           # scalar
            weight_sum      = torch.sum(weight_yn * ignore_mask)                  # scalar
            loss_out_grad   = mean_out_grad / (weight_sum + 1e-12)                # scalar
            smooth_loss_grad = grad_loss_fp32 * self.label_smoothing / num_classes / (weight_sum + 1e-12)  # scalar
            loss_out_grad    = loss_out_grad.unsqueeze(-1)                        # [1]
            smooth_loss_grad = smooth_loss_grad.unsqueeze(-1)                     # [1]

        elif self.reduction == "sum":
            sum_out_grad    = grad_loss_fp32 * (1.0 - self.label_smoothing)            # scalar
            loss_out_grad   = sum_out_grad.unsqueeze(-1)                          # [1]
            smooth_loss_grad = grad_loss_fp32 * self.label_smoothing / num_classes       # scalar
            smooth_loss_grad = smooth_loss_grad.unsqueeze(-1)                     # [1]

        else:  # "none"
            none_out_grad   = grad_loss_fp32 * (1.0 - self.label_smoothing)            # [N]
            loss_out_grad   = none_out_grad                                       # [N]
            smooth_loss_grad = grad_loss_fp32 * self.label_smoothing / num_classes     # [N]

        # 4. 应用 ignore mask
        loss_out_grad   = loss_out_grad   * ignore_mask        # [N] 或 [1]
        smooth_loss_grad = smooth_loss_grad * ignore_mask      # [N] 或 [1]

        # 5. 乘以 weight_yn
        nll_loss_grad = loss_out_grad * weight_yn              # [N]

        # 6. 计算 softmax 概率分支
        log_softmax_probs_grad_loss_out_sub_part = torch.exp(log_softmax_fp32) * nll_loss_grad.unsqueeze(-1)  # [N, C]

        # 7. 构造 one-hot 并填充
        predictions_grad_loss_out = torch.zeros(batch_size, num_classes, dtype=torch.float32, device=log_softmax.device)
        predictions_grad_loss_out.scatter_(1, target_fp32.unsqueeze(-1), nll_loss_grad.unsqueeze(-1))

        # 8. 最终梯度
        grad_input = log_softmax_probs_grad_loss_out_sub_part - predictions_grad_loss_out  # [N, C]

        # 9. 平滑项梯度
        if self.label_smoothing > 0:
            smooth_grad = smooth_loss_grad.unsqueeze(-1) * torch.ones_like(log_softmax_fp32)  # [N, C]
            grad_input += smooth_grad

        # ---------------- 转回原始 dtype ----------------
        return [grad_input.to(log_softmax.dtype)]


class ModelNew(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor], ignore_index: int, label_smoothing: float,
                 reduction: str, lse_square_scale_for_zloss: float):
        super(ModelNew, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.lse_square_scale_for_zloss = lse_square_scale_for_zloss

    def forward(self, grad_loss: torch.Tensor, log_prob: torch.Tensor, target: torch.Tensor,
                grad_zloss: Optional[torch.Tensor], lse_for_zloss: Optional[torch.Tensor]) -> List[torch.Tensor]:
        return kernel_gen_ops.cross_entropy_loss_grad(
            grad_loss,
            log_prob,
            target,
            self.weight,
            grad_zloss,
            lse_for_zloss,
            self.reduction,
            self.ignore_index,
            self.label_smoothing,
            self.lse_square_scale_for_zloss
        )