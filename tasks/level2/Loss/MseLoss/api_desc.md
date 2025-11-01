# aclnnMseLoss

## 功能描述

### 算子功能
该AscendC算子用于计算预测值 (`predict`) 与目标值 (`label`) 之间的均方误差 (Mean Squared Error, MSE)。MSE是回归任务中常用的损失函数。与 PyTorch 中的 `torch.nn.MSELoss` 类似，本算子通过 `reduction` 参数控制输出的计算方式。`reduction` 参数决定了是否对逐元素的损失进行缩减以及如何缩减，主要有三个选项：

-   **‘none’**: 不进行约简，返回每个元素的损失。
-   **‘mean’**: 返回所有元素损失的均值。
-   **‘sum’**: 返回所有元素损失的总和。

### 计算公式
根据 `reduction` 参数的设置，计算公式如下：

1.  当 `reduction='none'` 时：
    计算公式为（逐元素计算）：
    $$
    y = (predict - label)^2
    $$
    其中，$predict$ 是预测值张量，$label$ 是目标值张量，$y$ 是输出的逐元素损失张量。

2.  当 `reduction='mean'` 时：
    计算公式为：
    $$
    y = \text{mean}((predict - label)^2)
    $$
    其中，$y$ 是所有元素损失的平均值（标量）。

3.  当 `reduction='sum'` 时：
    计算公式为：
    $$
    y = \text{sum}((predict - label)^2)
    $$
    其中，$y$ 是所有元素损失的总和（标量）。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.mse_loss()` 函数形式提供：

```python
def mse_loss(predict, label, reduction='mean'):
    """
    计算预测值和目标值之间的均方误差损失。
    
    参数:
        predict (Tensor): 输入的预测值张量。Device侧的张量，数据格式支持ND。
        label (Tensor): 输入的目标值张量，形状应与 predict 张量相同。Device侧的张量，数据类型与 predict 一致，
                       数据格式支持ND。
        reduction (str, 可选): 指定应用于输出的缩减类型：
                               - 'none': 不应用缩减，返回逐元素的损失。
                               - 'mean': 输出的损失将是所有元素的平均值 (默认)。
                               - 'sum': 输出的损失将是所有元素的总和。
        
    返回:
        Tensor: 计算得到的损失张量。如果 reduction 为 'none'，则形状与输入相同；
               否则，为一个标量张量。数据类型与输入一致。
    
    注意:
        - predict 和 label 张量必须具有相同的形状和数据类型。
        - 张量数据格式支持ND。
        - reduction 参数必须是 'none'、'mean' 或 'sum' 之一。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
predict = torch.randn(8, 1024, dtype=torch.float32)
label = torch.randn(8, 1024, dtype=torch.float32)

# 使用gelu执行计算
result = kernel_gen_ops.mse_loss(predict, label, reduction='mean')
```

## 约束与限制

- predict，label 数据格式只支持ND。
- reduction 的数据类型只支持STRING，数据格式只支持SCALE。