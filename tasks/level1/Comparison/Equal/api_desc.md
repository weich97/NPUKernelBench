# aclnnEqual

## 功能描述

### 算子功能
Equal算子提供比较相等的功能。Equal算子的主要功能是对输入的两个数值（或张量）进行逐元素比较，判断每个元素是否相等。在数学和工程领域中，相等比较是一个基础且常见的操作，被广泛应用于图像处理、信号处理、逻辑运算等多个领域。Equal算子能够高效地处理批量数据的比较，支持浮点数和整数类型的输入。

### 计算公式
- 对于 `half` 和 `bfloat16_t` 类型

  $$
  y\_compute =
  \begin{cases} MIN\_ACCURACY\_FP16 & \text{if }  |x1−x2|>{ MIN\_ACCURACY\_FP16}  \\
   |x1−x2|& \text{if }  |x1−x2| <{ MIN\_ACCURACY\_FP16} 
  \end{cases}
  $$

  $$
  y=1-(y\_compute\times MAX\_MUL\_FP16)^2
  $$

  - 对于 `float` 类型

  $$
  y\_compute =
  \begin{cases} MIN\_ACCURACY\_FP32 & \text{if }  |x1−x2|>{ MIN\_ACCURACY\_FP32}  \\
   |x1−x2|& \text{if }  |x1−x2| <{ MIN\_ACCURACY\_FP32} 
  \end{cases}
  $$

  $$
  y=1-(y\_compute\times MAX\_MUL\_FP16)^3
  $$

   - 对于整数类型	

  $$
  \text{y} =
  \begin{cases} 0 & \text{if }  |x1−x2| >{ 1}  \\
  1-|x1−x2| & \text{if }  |x1−x2| <{ 1} 
  \end{cases}
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.equal()` 函数形式提供：


```python
def equal(input1, input2):
    """
    实现自定义张量元素级相等判断操作。

    参数:
        input1 (Tensor): 第一个输入张量，数据格式支持ND。
        input2 (Tensor): 第二个输入张量，与 input1 的形状和数据类型必须一致。

    返回:
        bool: 布尔值，表示两个张量的所有对应元素是否完全相等。
              - 如果所有元素相等，返回 True；
              - 否则返回 False。

    注意:
        - equal 不返回张量，而是返回 Python 的布尔类型（bool）；
        - 支持广播前形状必须一致，否则将抛出错误；
        - 对于浮点类型，精度要求严格（非近似相等）；
        - 使用场景：验证两个张量是否完全相等（常用于测试或结果对比）。
    """

```

## 使用案例

```
import torch
import kernel_gen_ops

# 创建两个张量
a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
b = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

# 检查它们是否完全相等
result = kernel_gen_ops.equal(a, b)
print(result)  # 输出: True

# 不相等的例子
c = torch.tensor([1.0, 2.0, 3.001], dtype=torch.float32)
print(kernel_gen_ops.equal(a, c))  # 输出: False
```
## 约束与限制

- 张量数据格式支持ND。


