from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    Reference implementation detail.
    """

    def __init__(self):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()

    def forward(self, var_ref: torch.Tensor, m_ref: torch.Tensor, v_ref: torch.Tensor, grad: torch.Tensor,
                step: torch.Tensor, max_grad_norm_ref: torch.Tensor,
                lr: float, beta1: float, beta2: float, weight_decay: float, eps: float, amsgrad: bool,
                maximize: bool) -> List[torch.Tensor]:
        """
        Reference implementation detail.

        Args:
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.
            Reference implementation detail.

        Returns:
            Reference implementation detail.
        """
        # Implementation note.
        dtype1, dtype2 = var_ref.dtype, grad.dtype

        if dtype1 != dtype2:
            grad = grad.to(dtype1)
            max_grad_norm_ref = max_grad_norm_ref.to(dtype1)

        # Implementation note.
        if maximize:
            grad = -grad

        step = step.item() + 1

        # Implementation note.
        # θ_t ← θ_t-1 - γλθ_t-1
        if weight_decay != 0:
            var_ref.mul_(1 - lr * weight_decay)

        # Implementation note.
        # m_t ← β_1 * m_t - 1 + (1 - β_1) * g_t
        m_ref.lerp_(grad, 1 - beta1)
        # v_t ← β_2 * v_t-1 + (1 - β_2) * g_t^2
        v_ref.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # Implementation note.
        beta1_t = beta1 ** step
        beta2_t = beta2 ** step
        # Implementation note.
        if amsgrad:
            torch.maximum(v_ref, max_grad_norm_ref, out=max_grad_norm_ref)
            # v̂_t ← v_t^max/(1 - β_2^t)
            v_hat = max_grad_norm_ref / (1 - beta2_t)
        else:
            # v̂_t ← v_t/(1 - β_2^t)
            v_hat = v_ref / (1 - beta2_t)

        # Implementation note.
        # m̂_t ← m_t/(1 - β_1^t)
        # θ_t ← θ_t - γm̂_t/(√v̂_t + ε)
        step_size = lr / (1 - beta1_t)
        denom = torch.sqrt(v_hat) + eps
        # θ_t ← θ_t - m_t * γ/(1 - β_1^t)/(√v̂_t + ε) = θ_t + (-step_size) * m_t / denom
        var_ref.addcdiv_(m_ref, denom, value=-step_size)

        # Implementation note.
        if amsgrad:
            return [var_ref, m_ref, v_ref, max_grad_norm_ref]
        else:
            return [var_ref, m_ref, v_ref]


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, var_ref: torch.Tensor, m_ref: torch.Tensor, v_ref: torch.Tensor, grad: torch.Tensor,
                step: torch.Tensor, max_grad_norm_ref: torch.Tensor,
                lr: float, beta1: float, beta2: float, weight_decay: float, eps: float, amsgrad: bool,
                maximize: bool) -> List[torch.Tensor]:
        import kernel_gen_ops
        return kernel_gen_ops.apply_adam_wv2(var_ref, m_ref, v_ref, grad, step, max_grad_norm_ref,
                                             lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)