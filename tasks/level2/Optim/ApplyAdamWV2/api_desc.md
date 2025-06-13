# aclnnApplyAdamWV2

## 功能描述

### 算子功能
`aclnnApplyAdamWV2` 实现了带权重衰减的Adam优化器(AdamW)功能，用于更新模型参数。

### 计算公式

AdamW 的实现步骤如下：

1. 首先应用权重衰减（与标准 Adam 的主要区别）：
$$\theta_{t} = \theta_{t-1} \cdot (1 - \gamma \lambda)$$

2. 更新一阶和二阶动量：
$$m_{t} = \beta_{1} \cdot m_{t-1} + (1 - \beta_{1}) \cdot g_{t}$$
$$v_{t} = \beta_{2} \cdot v_{t-1} + (1 - \beta_{2}) \cdot g_{t}^{2}$$

3. 计算偏差修正：
$$\hat{m}_{t} = \frac{m_{t}}{1 - \beta_{1}^{t}}$$
$$\hat{v}_{t} = \frac{v_{t}}{1 - \beta_{2}^{t}}$$

4. 当 amsgrad=True 时，使用历史二阶动量的最大值：
$$\hat{v}_{t} = \max(\hat{v}_{t-1}, \frac{v_{t}}{1 - \beta_{2}^{t}})$$

5. 更新参数：
$$\theta_{t} = \theta_{t} - \frac{\gamma}{1 - \beta_{1}^{t}} \cdot \frac{\hat{m}_{t}}{\sqrt{\hat{v}_{t}} + \epsilon}$$

其中:
- $\theta_t$ 是模型参数
- $m_t$ 和 $v_t$ 是一阶和二阶动量
- $\hat{m}_t$ 和 $\hat{v}_t$ 是偏差修正后的动量
- $g_t$ 是参数梯度
- $\gamma$ 是学习率
- $\beta_1$ 和 $\beta_2$ 是动量衰减系数
- $\lambda$ 是权重衰减系数
- $\epsilon$ 是数值稳定性常数
- $t$ 是当前迭代次数

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.apply_adam_wv2()` 函数形式提供：

```python
def apply_adam_wv2(var_ref, m_ref, v_ref, grad, step, max_grad_norm_ref, 
                  lr, beta1, beta2, weight_decay, eps, amsgrad, maximize):
    """
    实现AdamW优化器功能。
    
    参数:
        var_ref (Tensor): 待更新的权重参数，Device侧张量，支持非连续的Tensor，数据格式支持ND。
        m_ref (Tensor): 一阶动量，与var_ref形状和数据类型相同。
        v_ref (Tensor): 二阶动量，与var_ref形状和数据类型相同。
        grad (Tensor): 梯度数据，与var_ref形状和数据类型相同。
        step (Tensor): 当前迭代次数，Device侧张量，元素个数为1。
        max_grad_norm_ref (Tensor): 二阶动量的最大值，当amsgrad为True时必须提供，
                                           形状和数据类型与var_ref相同。
        lr (float): 学习率。
        beta1 (float): 一阶动量衰减系数。
        beta2 (float): 二阶动量衰减系数。
        weight_decay (float): 权重衰减系数。
        eps (float): 数值稳定性常数，防止除零。
        amsgrad (bool): 是否使用AMSGrad变体。
        maximize (bool): 是否最大化参数（即反向梯度）。
        
    返回:
        List[Tensor]: 包含更新后的参数和动量值的列表:
        - 若amsgrad为True: [var_ref_updated, m_ref_updated, v_ref_updated, max_grad_norm_updated]
        - 若amsgrad为False: [var_ref_updated, m_ref_updated, v_ref_updated]
    
    注意:
        - 张量数据格式支持ND
        - 支持非连续的Tensor
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建模型参数和相关状态
var = torch.randn(10, 5, dtype=torch.bfloat16)
m = torch.zeros_like(var)
v = torch.zeros_like(var)
max_grad = torch.zeros_like(var)  # 当amsgrad=True时需要
grad = torch.randn_like(var)  # 梯度
step = torch.tensor([1], dtype=torch.int64)  # 迭代次数

# AdamW优化器参数
lr = 0.01
beta1 = 0.9
beta2 = 0.99
weight_decay = 5e-3
eps = 1e-6
amsgrad = False
maximize = False

# 使用apply_adam_wv2执行参数更新
result = kernel_gen_ops.apply_adam_wv2(var, m, v, grad, step, max_grad,
                                      lr, beta1, beta2, weight_decay, 
                                      eps, amsgrad, maximize)

# 获取更新后的变量
var_updated, m_updated, v_updated, max_grad_updated = result
```

## 约束与限制

- 输入张量var_ref、m_ref、v_ref的数据类型必须一致。
- step张量的元素个数必须为1。