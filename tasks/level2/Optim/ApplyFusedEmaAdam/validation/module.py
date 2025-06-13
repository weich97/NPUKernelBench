from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    实现FusedEmaAdam融合优化器功能的模型。
    """

    def __init__(self, lr=0.001, emaDecay=0.999, beta1=0.9, beta2=0.999,
                 eps=1e-8, mode=0, biasCorrection=True, weightDecay=0.0):
        """
        初始化模型。
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
        实现Adam+EMA优化计算逻辑

        参数:
            grad: 梯度
            varRef: 变量引用
            mRef: 一阶动量引用
            vRef: 二阶动量引用
            sRef: EMA平均值引用
            step: 当前步数

        返回:
            更新后的[变量, 一阶动量, 二阶动量, EMA平均值]
        """
        # 判断是否需要转换数据类型
        original_dtype = grad.dtype
        need_cast = original_dtype == torch.float16 or original_dtype == torch.bfloat16

        # 如果需要，将输入转换为float32
        if need_cast:
            grad = grad.to(torch.float32)
            varRef = varRef.to(torch.float32)
            mRef = mRef.to(torch.float32)
            vRef = vRef.to(torch.float32)
            sRef = sRef.to(torch.float32)

        # 计算beta1和beta2的修正系数
        if self.biasCorrection:
            beta1_correction = 1.0 - self.beta1 ** step
            beta2_correction = 1.0 - self.beta2 ** step
        else:
            beta1_correction = 1.0
            beta2_correction = 1.0

        # 根据mode计算修正梯度
        if self.mode == 0:
            grad_ = grad + self.weightDecay * varRef
        elif self.mode == 1:
            grad_ = grad

        # 更新一阶动量
        m_ = self.beta1 * mRef + (1 - self.beta1) * grad_

        # 更新二阶动量
        v_ = self.beta2 * vRef + (1 - self.beta2) * grad_ * grad_

        # 应用偏差修正
        next_m = m_ / beta1_correction
        next_v = v_ / beta2_correction

        # 计算更新值的分母
        denom = torch.sqrt(next_v) + self.eps

        # 根据mode计算更新值
        if self.mode == 0:
            update = next_m / denom
        elif self.mode == 1:
            update = next_m / denom + self.weightDecay * varRef

        # 更新变量
        var_ = varRef - self.lr * update

        # 更新EMA平均值
        s_ = self.emaDecay * sRef + (1 - self.emaDecay) * var_

        # 如果需要，将结果转换回原始数据类型
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