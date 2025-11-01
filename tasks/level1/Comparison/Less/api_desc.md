# aclnnLess

## 功能描述

### 算子功能
返回一个和输入张量同样形状大小的新张量，它的每一个元素是输入的两个张量对应元素比较大小（第一个元素是否小于第二个元素）的结果。

### 计算公式
$$
x1 = [x1_{0}, x1_{1}, ... x1_{n-1}], x2 = [x2_{0}, x2_{1}, ... x2_{n-1}]\\
y = [y_{0}, y_{1}, ... y_{n-1}]\\
$$

$$
y_i = \begin{cases}
True, & \text{if } x1_i < x2_i \\
False, & \text{otherwise}
\end{cases} (i = 0, 1, ..., n - 1)
$$

### AscendC 实现范式注意事项

在 AscendC 中，比较类算子普遍采用一套 `Compare -> Duplicate -> Select -> Cast` 的标准指令组合。这是一种将硬件的逻辑比较结果“物化”为标准张量的通用方法。

其核心逻辑是：
1.  **`Compare`**：执行比较，生成一个内部的逻辑掩码（mask），标记出真/假位置。
2.  **`Duplicate` 与 `Select`**：协同使用该掩码，从预设的 `1` (真) 和 `0` (假) 中进行选择，从而将逻辑掩码转换为一个内容为 `0` 和 `1` 的实体张量。
3.  **`Cast`**：最后将此张量转换为最终需要的输出数据类型（如`int8_t`）。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.less()` 函数形式提供：

```python
def less(tensor1, tensor2):
    """
    实现自定义张量元素级小于判断操作。

    参数:
        input1 (Tensor): 第一个输入张量，数据格式支持ND。
        input2 (Tensor): 第二个输入张量，与 input1 的形状和数据类型必须一致。

    返回:
        Tensor(bool): 布尔张量，形状与输入相同。
                      - 如果 input1 的元素小于 input2 的对应元素，返回 True；
                      - 否则返回 False。

    注意:
        - less 返回的是布尔类型的张量，而不是单个布尔值；
        - 输入张量的形状必须一致，否则将抛出错误；
        - 对于浮点类型比较，直接进行逐元素数值比较（非近似）；
        - 使用场景：可用于条件判断、掩码(mask)生成等。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建第一个输入张量
tensor1 = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

# 创建第二个输入张量
tensor2 = torch.tensor([2.0, 1.0, 4.0], dtype=torch.float32)

# 使用 less 执行计算
result = kernel_gen_ops.less(tensor1, tensor2)
```
## 约束与限制

- 张量数据格式支持ND。


