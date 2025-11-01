# aclnnLessEqual

## 功能描述

### 算子功能
逐元素比较两个向量，当 `x1 <= x2` 时，输出 `true` (1)，否则输出 `false` (0)。

### 计算公式
$$y = x1 <= x2$$

### AscendC 实现范式注意事项

在 AscendC 中，比较类算子普遍采用一套 `Compare -> Duplicate -> Select -> Cast` 的标准指令组合。这是一种将硬件的逻辑比较结果“物化”为标准张量的通用方法。

其核心逻辑是：
1.  **`Compare`**：执行比较，生成一个内部的逻辑掩码（mask），标记出真/假位置。
2.  **`Duplicate` 与 `Select`**：协同使用该掩码，从预设的 `1` (真) 和 `0` (假) 中进行选择，从而将逻辑掩码转换为一个内容为 `0` 和 `1` 的实体张量。
3.  **`Cast`**：最后将此张量转换为最终需要的输出数据类型（如`int8_t`）。


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

```python
import torch
import kernel_gen_ops


# 创建两个张量
input1 = torch.randint(-10, 10, 8, 2048, dtype=torch.float32)
input2 = torch.randint(-10, 10, 8, 2048, dtype=torch.float32)

# 元素级判断是否小于等于
result = kernel_gen_ops.less_equal(input1, input2)
print(result)

```
## 约束与限制

- 张量数据格式支持ND。


