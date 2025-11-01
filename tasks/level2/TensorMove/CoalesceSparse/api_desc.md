# aclnnCoalesceSparse

## 功能描述

### 算子功能
`aclnnCoalesceSparse` 算子用于合并稀疏张量中具有相同索引的元素，并将它们的值进行累加。它接收稀疏张量的长度信息、唯一索引、索引以及对应的值，并输出合并后的新索引和新值。

### 计算公式

该算子主要涉及对输入索引进行排序和比较，对于具有相同索引的条目，将其对应的值进行累加。具体的数学公式取决于稀疏张量的表示方式和合并的具体逻辑，通常不涉及简单的代数公式。

给定输入：
- `uniqueLen`: 表示每个维度中唯一索引的数量。
- `uniqueIndices`: 包含所有维度的唯一索引值。
- `indices`: 稀疏张量的索引，每一行代表一个元素的坐标。
- `values`: 稀疏张量的值，与 `indices` 中的每一行对应。

输出：
- `newIndices`: 合并后的唯一索引。
- `newValues`: 与 `newIndices` 对应的累加后的值。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.coalesce_sparse()` 函数形式提供：

```python
def coalesce_sparse(unique_len: Tensor, unique_indices: Tensor, indices: Tensor, values: Tensor) -> List[Tensor]:
    """
    合并稀疏张量中具有相同索引的元素。

    参数:
        unique_len (Tensor): 1 维张量，表示每个维度中唯一索引的数量，数据类型支持 torch.int32或torch.int64。
        unique_indices (Tensor): 1 维张量，包含所有维度的唯一索引值，数据类型支持 torch.int32或torch.int64。
        indices (Tensor): 2 维张量，稀疏张量的索引，形状为 [N, D]且D必须小于64，每一行代表一个元素的坐标，数据类型支持 torch.int32或torch.int64。
        values (Tensor): N 维张量，稀疏张量的值，与 `indices` 中的每一行对应，数据类型支持 torch.float16, torch.int32, torch.float。

    返回:
        List[Tensor]: 包含两个输出张量 (newIndices, newValues)。
            - newIndices (Tensor): 合并后的唯一索引，形状可能与输入 `indices` 不同，数据类型为 torch.int32。
            - newValues (Tensor): 与 `newIndices` 对应的累加后的值，数据类型与输入 `values` 相同。

    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 构造输入张量
unique_len = torch.tensor([2, 2], dtype=torch.int32)
unique_indices = torch.tensor([0, 1, 0, 1], dtype=torch.int32)
indices = torch.tensor([[0, 0], [1, 1], [0, 0]], dtype=torch.int32)
values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16)

# 执行 CoalesceSparse 操作
new_indices, new_values = kernel_gen_ops.coalesce_sparse(unique_len, unique_indices, indices, values)

```

## 约束与限制

假设indices的shape为[6,8], values的shape为[6, 8, 8], indices按dim=0做max的结果为[6, 6, 6, 5, 6, 6, 5, 6], 将该结果的倒序得到[6, 5, 6, 6, 5, 6, 6, 6], 然后将其视为shape，计算stride=[194400, 38880, 6480, 1080, 216, 36, 6, 1], 则对所有的0≤n<6都需要满足$\sum_{m=0}^{m=7} indices[n][m] * stride[m] \lt INT64\_MAX$。