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

    def __init__(self, lr=0.001, emaDecay=0.999, beta1=0.9, beta2=0.999,
                 eps=1e-8, mode=0, biasCorrection=True, weightDecay=0.0):
        """
        Reference implementation detail.
        """
        super(Model, self).__init__()
        self.lr = lr
        self.emaDecay = emaDecay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mode = mode
        self.biasCorrection = biasCorrection
        self.weightDecay = weightDecay

    def forward(self, grad: torch.Tensor, varRef: torch.Tensor, mRef: torch.Tensor,
                vRef: torch.Tensor, sRef: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        """
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
        """
        # Implementation note.
        original_dtype = grad.dtype
        need_cast = original_dtype == torch.float16 or original_dtype == torch.bfloat16

        # Implementation note.
        if need_cast:
            grad = grad.to(torch.float32)
            varRef = varRef.to(torch.float32)
            mRef = mRef.to(torch.float32)
            vRef = vRef.to(torch.float32)
            sRef = sRef.to(torch.float32)

        # Implementation note.
        if self.biasCorrection:
            beta1_correction = 1.0 - self.beta1 ** step
            beta2_correction = 1.0 - self.beta2 ** step
        else:
            beta1_correction = 1.0
            beta2_correction = 1.0

        # Implementation note.
        if self.mode == 0:
            grad_ = grad + self.weightDecay * varRef
        elif self.mode == 1:
            grad_ = grad

        # Implementation note.
        m_ = self.beta1 * mRef + (1 - self.beta1) * grad_

        # Implementation note.
        v_ = self.beta2 * vRef + (1 - self.beta2) * grad_ * grad_

        # Implementation note.
        next_m = m_ / beta1_correction
        next_v = v_ / beta2_correction

        # Implementation note.
        denom = torch.sqrt(next_v) + self.eps

        # Implementation note.
        if self.mode == 0:
            update = next_m / denom
        elif self.mode == 1:
            update = next_m / denom + self.weightDecay * varRef

        # Implementation note.
        var_ = varRef - self.lr * update

        # Implementation note.
        s_ = self.emaDecay * sRef + (1 - self.emaDecay) * var_

        # Implementation note.
        if need_cast:
            var_ = var_.to(original_dtype)
            m_ = m_.to(original_dtype)
            v_ = v_.to(original_dtype)
            s_ = s_.to(original_dtype)

        return [var_, m_, v_, s_]


class ModelNew(nn.Module):
    def __init__(self, lr=0.001, emaDecay=0.999, beta1=0.9, beta2=0.999,
                 eps=1e-8, mode=0, biasCorrection=True, weightDecay=0.0):
        super(ModelNew, self).__init__()
        self.lr = lr
        self.emaDecay = emaDecay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mode = mode
        self.biasCorrection = biasCorrection
        self.weightDecay = weightDecay

    def forward(self, grad: torch.Tensor, varRef: torch.Tensor, mRef: torch.Tensor,
                vRef: torch.Tensor, sRef: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.apply_fused_ema_adam(grad, varRef, mRef, vRef, sRef, step,
                                                    self.lr, self.emaDecay, self.beta1, self.beta2,
                                                    self.eps, self.mode, self.biasCorrection, self.weightDecay)