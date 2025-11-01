# aclnnRmsNorm

## 功能描述
### 算子功能
对输入张量执行均方根归一化（RMS Norm）操作。

### 计算公式
$$
y = {{x}\over\sqrt {Mean(x^2)+eps}} * \gamma
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.rms_norm()` 函数形式提供：

```python
def rms_norm(x, gamma, epsilon):
    """
    对输入张量执行均方根归一化（RMS Norm）操作。

    参数:
        x (Tensor): 输入Device侧张量。
        gamma (Tensor): 缩放参数张量，与x需要norm的维度的维度值相同，用于RMS归一化。
        epsilon (double): 公式中的输入eps，添加到分母中的值，以确保数值稳定。

    返回:
        List[Tensor]: 一个新的Device侧张量，表示输入张量RMS归一化的结果。
    """
```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量和参数
x = torch.tensor([[1.0, -2.0], [3.0, 4.0]], dtype=torch.float)
gamma = torch.tensor([1.0, 1.0], dtype=torch.float)
epsilon = 1e-6

# 使用 rms_norm 对输入张量执行均方根归一化（RMS Norm）操作
y = kernel_gen_ops.rms_norm(x, gamma, epsilon)
```

## 约束与限制
- **功能维度**
  * x, gamma支持：FLOAT32、FLOAT16、BFLOAT16。
  * 数据格式支持：ND。
  * 张量形状维度不高于8维。