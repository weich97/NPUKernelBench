from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    实现adamW优化器功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, var_ref: torch.Tensor, m_ref: torch.Tensor, v_ref: torch.Tensor, grad: torch.Tensor,
                step: torch.Tensor, max_grad_norm_ref: torch.Tensor,
                lr: float, beta1: float, beta2: float, weight_decay: float, eps: float, amsgrad: bool,
                maximize: bool) -> List[torch.Tensor]:
        """
        实现adamW优化器功能。

        Args:
            var_ref: 待计算的权重输入同时也是输出，公式中的theta
            m_ref: adamw优化器中m参数，公式中的m
            v_ref: adamw优化器中v参数，公式中的v
            max_grad_norm_ref: 保存v参数的最大值，公式中的v_max
            grad: 梯度数据，公式中的gt
            step: 迭代次数，公式中的t
            lr: 学习率，公式中的gamma
            beta1: beta1参数
            beta2: beta2参数
            weight_decay: 权重衰减系数, 公式中的λ
            eps: 防止除数为0, 公式中的ε
            amsgrad: 是否使用算法的AMSGrad变体
            maximize: 是否最大化参数

        Returns:
            更新后的var_ref, m_ref, v_ref和max_grad_norm_ref列表
        """
        # 处理不同的数据类型
        dtype1, dtype2 = var_ref.dtype, grad.dtype

        if dtype1 != dtype2:
            grad = grad.to(dtype1)
            max_grad_norm_ref = max_grad_norm_ref.to(dtype1)

        # 如果maximize为True，则反转梯度
        if maximize:
            grad = -grad

        step = step.item() + 1

        # AdamW: 首先应用权重衰减到参数，直接从参数中减去
        # θ_t ← θ_t-1 - γλθ_t-1
        if weight_decay != 0:
            var_ref.mul_(1 - lr * weight_decay)

        # 更新一阶和二阶动量
        # m_t ← β_1 * m_t - 1 + (1 - β_1) * g_t
        m_ref.lerp_(grad, 1 - beta1)
        # v_t ← β_2 * v_t-1 + (1 - β_2) * g_t^2
        v_ref.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        # 修正偏差
        beta1_t = beta1 ** step
        beta2_t = beta2 ** step
        # 处理AMSGrad变体
        if amsgrad:
            torch.maximum(v_ref, max_grad_norm_ref, out=max_grad_norm_ref)
            # v̂_t ← v_t^max/(1 - β_2^t)
            v_hat = max_grad_norm_ref / (1 - beta2_t)
        else:
            # v̂_t ← v_t/(1 - β_2^t)
            v_hat = v_ref / (1 - beta2_t)

        # 计算参数更新值
        # m̂_t ← m_t/(1 - β_1^t)
        # θ_t ← θ_t - γm̂_t/(√v̂_t + ε)
        step_size = lr / (1 - beta1_t)
        denom = torch.sqrt(v_hat) + eps
        # θ_t ← θ_t - m_t * γ/(1 - β_1^t)/(√v̂_t + ε) = θ_t + (-step_size) * m_t / denom
        var_ref.addcdiv_(m_ref, denom, value=-step_size)

        # 返回结果
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