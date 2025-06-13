# ApplyFusedEmaAdam

## 功能描述

## 功能描述

- **算子功能**：实现FusedEmaAdam融合优化器功能。

- **计算公式**：

  $$
  (correction_{\beta_1},correction_{\beta_2},)=\begin{cases}
  (1,1),&biasCorrection=False\\
  (1-\beta_1^{step},1-\beta_2^{step}),&biasCorrection=True
  \end{cases}
  $$
  
  $$
  grad=\begin{cases}
  grad+weightDecay*var,&mode=0\\
  grad,&mode=1
  \end{cases}
  $$
  
  $$
  m_{out}=\beta_1*m+(1-\beta_1)*grad
  $$

  $$
  v_{out}=\beta_2*v+(1-\beta_2)*grad^2
  $$

  $$
  m_{next}=m_{out}/correction_{\beta_1}
  $$

  $$
  v_{next}=v_{out}/correction_{\beta_2}
  $$

  $$
  denom=\sqrt{v_{next}}+eps
  $$

  $$
  update=\begin{cases}
  m_{next}/denom,&mode=0\\
  m_{next}/denom+weightDecay*var,&mode=1
  \end{cases}
  $$

  $$
  var_{out}=var-lr*update
  $$

  $$
  s_{out}=emaDecay*s+(1-emaDecay)*var_{out}
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.apply_fused_ema_adam()` 函数形式提供：

```python
def apply_fused_ema_adam(var_ref, m_ref, v_ref, s_ref, grad, step, 
                         lr, ema_decay, beta1, beta2, eps, mode, bias_correction, weight_decay):
    """
    实现FusedEmaAdam优化器功能，结合指数移动平均（EMA）和Adam优化器。

    参数:
        var_ref (Tensor): 待更新的权重参数，Device侧张量，支持非连续的Tensor，数据格式支持ND。
        m_ref (Tensor): 一阶动量，与var_ref形状和数据类型相同。
        v_ref (Tensor): 二阶动量，与var_ref形状和数据类型相同。
        s_ref (Tensor): EMA权重，与var_ref形状和数据类型相同。
        grad (Tensor): 梯度数据，与var_ref形状和数据类型相同。
        step (Tensor): 当前迭代次数，Device侧张量，元素个数为1。
        lr (float): 学习率。
        ema_decay (float): 指数移动平均（EMA）的衰减速率。
        beta1 (float): 一阶动量衰减系数。
        beta2 (float): 二阶动量衰减系数。
        eps (float): 数值稳定性常数，防止除零。
        mode (int): 控制应用L2正则化还是权重衰减，1为AdamW，0为L2。
        bias_correction (bool): 是否进行偏差校正，True表示校正，False表示不校正。
        weight_decay (float): 权重衰减系数。

    返回:
        List[Tensor]: 包含更新后的参数和动量值的列表: [var_ref_updated, m_ref_updated, v_ref_updated, s_ref_updated]

    注意:
        - 张量数据格式支持ND。
        - 支持非连续的Tensor。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建模型参数和相关状态
var = torch.randn(8, 2048, dtype=torch.float16)
m = torch.zeros_like(var)
v = torch.zeros_like(var)
s = torch.zeros_like(var)  # EMA权重
grad = torch.randn_like(var)  # 梯度
step = torch.tensor([10], dtype=torch.int64)  # 迭代次数

# FusedEmaAdam优化器参数
lr = 0.001
ema_decay = 0.999
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
mode = 0  # 1为AdamW，0为L2
bias_correction = True
weight_decay = 0.0

# 使用apply_fused_ema_adam执行参数更新
result = kernel_gen_ops.apply_fused_ema_adam(var, m, v, s, grad, step,
                                             lr, ema_decay, beta1, beta2, 
                                             eps, mode, bias_correction, weight_decay)

# 获取更新后的变量
var_updated, m_updated, v_updated, s_updated = result
```

## 约束与限制
无