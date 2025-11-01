# aclnnReverseSequence

## 功能描述

### 算子功能
`aclnnReverseSequence` 沿指定的序列维度（`seqDim`）反转每个批次（由 `batchDim` 指定）中变长序列的元素。序列的实际长度由 `seqLengths` 指定。

### 计算公式

对于一个输入张量 $x$ 和序列长度张量 $seqLengths$，沿维度 `seqDim`，对于批次维度 `batchDim` 中的每个批次 $b$，将前 $seqLengths[b]$ 个元素进行反转。其余元素保持不变。

例如，如果 $x$ 的形状是 $[B, T, D]$，`seqDim=1`，`batchDim=0`，且 $seqLengths = [t_1, t_2, ..., t_B]$，那么对于每个 $i \in [0, B-1]$，`x[i, 0:t_i, :]` 将沿时间维度（维度 1）反转。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.reverse_sequence()` 函数形式提供：

```python
def reverse_sequence(x: Tensor, seq_lengths: Tensor, seq_dim: int = 0, batch_dim: int = 1) -> Tensor:
    """
    沿指定的序列维度反转每个批次中变长序列的元素。

    参数:
        x (Tensor): 输入 Device 侧张量。支持的数据类型包括：
                    torch.float32、torch.float16、torch.bfloat16、torch.float64、torch.int8、torch.int16、torch.int32、torch.int64、torch.uint8、torch.uint16、torch.bool、torch.complex64、torch.complex128。
                    支持非连续 Tensor，shape 维度不超过 8 维，数据格式支持 ND。
        seq_lengths (Tensor): 一维的 Device 侧 int32或int64 张量，包含每个批次的序列长度。其大小必须等于输入张量 `x` 在 `batch_dim` 上的大小。
        seq_dim (int): 指定进行反转的序列维度。
        batch_dim (int): 指定批次维度。默认为 0。

    返回:
        Tensor: 输出张量，与输入张量 `x` 具有相同的形状和数据类型。

    注意:
        - `seqLengths` 的大小必须与输入张量在 `batch_dim` 上的大小一致。
        - `seqDim` 和 `batchDim` 必须是有效的维度索引且不能相等。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 构造输入张量和序列长度
x = torch.randn(3, 5, 7, dtype=torch.float)
seq_lengths = torch.tensor([2, 4, 1], dtype=torch.int64)
seq_dim = 1
batch_dim = 0

# 执行 ReverseSequence 操作
y = kernel_gen_ops.reverse_sequence(x, seq_lengths, seq_dim, batch_dim)
```

### 约束与限制
seqLengths 必须是一维张量。
seqLengths 的大小必须等于输入张量在 batch_dim 上的大小。
seqDim 和 batchDim 必须是有效的维度索引且不能相等。