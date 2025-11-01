# module.py
from typing import List, Optional, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor], ignore_index: int, label_smoothing: float,
                 reduction: str, lse_square_scale_for_zloss: float, return_zloss: bool):
        super(Model, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.lse_square_scale_for_zloss = lse_square_scale_for_zloss
        self.return_zloss = return_zloss

    def forward(self, input_predictions: torch.Tensor, target_labels: torch.Tensor) -> List[torch.Tensor]:
        n, c = input_predictions.shape # batch, num_classes
        input_dtype = input_predictions.dtype

        # Convert predictions to float32 for internal calculation (common for stability)
        # 这一步可能会导致问题，因为最终 ZLoss 的输出需要匹配 input_dtype
        # 最好在NPU侧完成相应的类型转换，或者确保所有中间计算都使用兼容的类型
        # 在这里，我们先让 predictions_fp32 继续使用，但会在 ZLoss 相关输出时进行类型转换
        predictions_fp32 = input_predictions.to(torch.float32) 

        # Handle weight: if None, create ones
        if self.weight is None:
            weight_fp32 = torch.ones((c,), dtype=torch.float32, device=predictions_fp32.device)
        else:
            weight_fp32 = self.weight.to(torch.float32)

        # 1. Calculate LogSoftmax Probabilities (logProbOut)
        predictions_max = torch.max(predictions_fp32, dim=1, keepdim=True)[0]
        
        # Calculate LogSumExp (LSE)
        lse = predictions_max + torch.log(torch.sum(torch.exp(predictions_fp32 - predictions_max), dim=1, keepdim=True))
        
        log_softmax_probs = predictions_fp32 - lse # This is logProbOut

        # 2. Calculate NLL Loss term
        # unsqueeze targets to (N, 1) to use gather
        nll_loss_terms = torch.gather(log_softmax_probs, 1, target_labels.unsqueeze(-1)).squeeze(-1) # NLL for correct class

        # Apply weight to NLL loss
        weight_for_targets = torch.gather(weight_fp32, 0, target_labels)
        loss_out_unreduced = -nll_loss_terms * weight_for_targets # This is the "none" reduction before ignore_index

        # 3. Handle ignore_index
        if self.ignore_index >= 0:
            # Create a mask: True for valid targets, False for ignored targets
            ignore_mask = (target_labels != self.ignore_index).float() # (N,)
            loss_out_unreduced = loss_out_unreduced * ignore_mask
        else:
            ignore_mask = torch.ones((n,), dtype=torch.float32, device=predictions_fp32.device)

        # 4. Calculate Label Smoothing Loss term (smooth_loss)
        # Sum over classes, multiplied by weight (unsqueezed to broadcast over batch dim)
        smooth_loss_unreduced = -torch.sum(log_softmax_probs * weight_fp32.unsqueeze(0), dim=1, keepdim=False) # (N,)
        if self.ignore_index >= 0:
            smooth_loss_unreduced = smooth_loss_unreduced * ignore_mask

        # 5. Apply Reduction (lossOut)
        # Total sum of valid weights for 'mean' reduction denominator
        weight_after_mask_sum = torch.sum(weight_for_targets * ignore_mask, dim=-1, keepdim=False) # Scalar for mean

        # Base loss without smoothing
        base_loss_reduced = None
        if self.reduction == "mean":
            # Handle case where weight_after_mask_sum might be zero (all ignored)
            # Avoid division by zero, return 0 if sum is 0.
            base_loss_reduced = torch.sum(loss_out_unreduced, dim=-1, keepdim=False) / (weight_after_mask_sum + 1e-12)
        elif self.reduction == "sum":
            base_loss_reduced = torch.sum(loss_out_unreduced, dim=-1, keepdim=False)
        else: # "none"
            base_loss_reduced = loss_out_unreduced

        # Apply label smoothing to the reduced loss
        smoothed_term_reduced = None
        if self.reduction == "mean":
            smoothed_term_reduced = torch.sum(smooth_loss_unreduced, dim=-1, keepdim=False) / (weight_after_mask_sum + 1e-12) * self.label_smoothing / c
        elif self.reduction == "sum":
            smoothed_term_reduced = torch.sum(smooth_loss_unreduced, dim=-1, keepdim=False) * self.label_smoothing / c
        else: # "none"
            smoothed_term_reduced = smooth_loss_unreduced * self.label_smoothing / c

        loss_out = (1 - self.label_smoothing) * base_loss_reduced + smoothed_term_reduced

        # 6. Calculate Zloss and LseForZlossOut (if return_zloss is True)
        # --- START CHANGE ---
        # Ensure ZLoss outputs match input_dtype if return_zloss is True
        zloss_out_dtype = input_dtype if input_dtype in [torch.float16, torch.bfloat16] else torch.float32
        
        zloss_out = torch.zeros((1,), dtype=zloss_out_dtype, device=predictions_fp32.device) 
        lse_for_zloss_out = lse.squeeze(-1) # LSE is already computed, just squeeze for output shape
        
        if self.return_zloss:
            zloss_out = self.lse_square_scale_for_zloss * torch.mean(lse.pow(2))
            zloss_out = zloss_out.reshape(1) # Ensure scalar for `zlossOut`
        # --- END CHANGE ---

        # Return values (lossOut, logProbOut, zlossOut, lseForZlossOut)
        # Convert all outputs back to the input_dtype
        return [
            loss_out.to(input_dtype),
            log_softmax_probs.to(input_dtype),
            zloss_out.to(input_dtype), # Ensure this is converted to input_dtype
            lse_for_zloss_out.to(input_dtype) # Ensure this is converted to input_dtype
        ]


class ModelNew(nn.Module):
    def __init__(self, weight: Optional[torch.Tensor], ignore_index: int, label_smoothing: float,
                 reduction: str, lse_square_scale_for_zloss: float, return_zloss: bool):
        super(ModelNew, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        self.lse_square_scale_for_zloss = lse_square_scale_for_zloss
        self.return_zloss = return_zloss

    def forward(self, input_predictions: torch.Tensor, target_labels: torch.Tensor) -> List[torch.Tensor]:
        # The key is to pass the correct parameters for the custom operator
        # The `return_zloss` parameter needs to be correctly passed as a boolean.
        # Also ensure `lse_square_scale_for_zloss` is correctly passed.
        
        # The order of arguments to kernel_gen_ops.cross_entropy_loss must match the pybind.cpp function
        return kernel_gen_ops.cross_entropy_loss(
            input_predictions, 
            target_labels, 
            self.weight, 
            self.reduction, # string
            self.ignore_index, 
            self.label_smoothing, # double
            self.lse_square_scale_for_zloss, # double
            self.return_zloss # bool
        )