# aclnnFlashAttentionScoreWithLargeHeadDim

## 功能描述

### 算子功能
训练场景下，使用FlashAttention算法实现self-attention（自注意力）的计算。

### 计算公式

 $$
    attention\_out = Dropout(Softmax(Mask(scale*(pse+query*key^T), atten\_mask)), keep\_prob)*value
    $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.flash_attention_score_with_large_head_dim()` 函数形式提供：


```python
def flash_attention_score_with_large_head_dim(query, key, value, scale_value, head_num):
    """
    实现 FlashAttention 计算逻辑，支持大 Head 维度。

    参数:
        query (Tensor): 查询张量，形状为 [batch, seq_len_q, head_dim]。
        key (Tensor): 键张量，形状为 [batch seq_len_k, head_dim]。
        value (Tensor): 值张量，形状为 [batch, seq_len_k, head_dim_v]。
        scale_value (float, optional): 缩放因子，默认 1.0。
        head_num (int): 注意力头的数量。

    返回:
        Tensor: attention 输出张量，数据类型与输入一致。

    注意:
        - 输入张量需满足可广播规则；
        - 支持多维批次处理；
        - 推荐用于推理或训练阶段的高效注意力计算。
    """


```

## 使用案例

```
import torch
import kernel_gen_ops

shape = (1, 2048, 576)

query = torch.randn(shape, device=device, dtype=torch.float16) * 9 + 1  # 均匀分布[1,10)
key = torch.randn(shape, device=device, dtype=torch.float16) * 9 + 1
value = torch.randn(shape, device=device, dtype=torch.float16) * 9 + 1

scale_value = 0.0625
head_num = 576

grad = kernel_gen_ops.flash_attention_score_with_large_head_dim(query, key, value, scale_value, head_num)
```
## 约束与限制

- 张量数据格式支持ND。


