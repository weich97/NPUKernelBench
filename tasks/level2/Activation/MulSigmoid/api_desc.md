# aclnnMulSigmoid

## 功能描述

### 算子功能
返回两个输入张量经过 `mul->sigmoid` 计算后的结果张量。

### 计算公式
设输入张量为 $x1$ 和 $x2$，标量为 $t1$，$t2$，$t3$，则计算过程如下：
$$
tmp = \frac{1}{1 + e^{-x1 * t1}}
$$
$$
sel = 
\begin{cases}
tmp, & \text{if } tmp < t2 \\
2 * tmp, & \text{otherwise}
\end{cases}
$$
$$
res = sel * x2 * t3
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.mul_sigmoid()` 函数形式提供：

```python
def mul_sigmoid(x1, x2, t1, t2, t3):
    """
    实现自定义 MulSigmoid 操作。
    
    参数:
        x1 (Tensor): Device侧的aclTensor，公式中的输入x1，数据类型支持FLOAT16，数据格式支持ND。
        x2 (Tensor): Device侧的aclTensor，公式中的输入x2，数据类型支持FLOAT16，数据格式支持ND。
        t1 (Tensor): 标量，公式中的输入t1，数据类型支持FLOAT32。
        t2 (Tensor): 标量，公式中的输入t2，数据类型支持FLOAT32。
        t3 (Tensor): 标量，公式中的输入t3，数据类型支持FLOAT32。
        
    返回:
        Tensor: Device侧的aclTensor，公式中的输出out，数据类型支持FLOAT16，数据格式支持ND。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建第一个输入张量
x1 = torch.randn([25, 32*1024], dtype=torch.float16)

# 创建第二个输入张量
x2 = torch.randn([1, 256, 128], dtype=torch.float16)

# 创建标量张量
t1 = torch.tensor([0.3], dtype=torch.float16)
t2 = torch.tensor([0.1], dtype=torch.float16)
t3 = torch.tensor([0.8], dtype=torch.float16)

# 使用 mul_sigmoid 执行计算
result = kernel_gen_ops.mul_sigmoid(x1, x2, t1, t2, t3)
```
## 约束与限制

- 张量数据格式支持ND。


