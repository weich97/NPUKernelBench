# aclnnLessEqual

## 功能描述

### 算子功能
实现了当向量x1小于等于x2时，向量y的对应位置输出true，否则为false的功能。

### 计算公式
$$
  y = x1 <= x2
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.less_equal()` 函数形式提供：


```python
def less_equal(input1, input2):
    """
    实现自定义张量元素级小于等于判断操作。

    参数:
        input1 (Tensor): 第一个输入张量，数据格式支持 ND。
        input2 (Tensor): 第二个输入张量，形状与 input1 可广播。

    返回:
        Tensor: 一个布尔类型张量，形状为输入张量广播后的形状。
                - 对于每个对应元素，如果 input1 <= input2，则结果为 True，否则为 False。

    注意:
        - 返回的是布尔张量（每个元素是否满足条件）；
        - 支持广播机制（形状兼容的情况下自动对齐）；
        - 不要求两个张量 shape 完全一致，只要可以广播即可；
    """

```

## 使用案例

```
import torch
import kernel_gen_ops


# 创建两个张量
a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
b = torch.tensor([1.5, 2.0, 2.5], dtype=torch.float32)

# 元素级判断是否小于等于
result = kernel_gen_ops.less_equal(a, b)
print(result)  # 输出: tensor([True, True, False])

# 广播示例
c = torch.tensor([2.0], dtype=torch.float32)
print(kernel_gen_ops.less_equal(a, c))  # 输出: tensor([True, True, False])
```
## 约束与限制

- 张量数据格式支持ND。


