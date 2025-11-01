# aclnnMla

## 功能描述

### 算子功能
该Ascend C算子用于执行一种高效的自注意力机制——Paged Attention，该机制专为大规模语言模型（LLM）的推理场景设计。它通过一个非连续的、分页式的Key-Value缓存（KV Cache）来管理内存，显著降低了显存碎片，提高了内存利用率。

该算子支持分组查询注意力（Grouped-Query Attention, GQA），允许多个查询头（Query Head）共享同一组键/值头（Key/Value Head），从而在保持模型性能的同时，大幅减少KV缓存的内存占用和带宽需求。

算子的核心功能是，根据给定的`query`张量、分页的`kv_cache`以及用于索引物理内存块的`block_tables`，计算出加权的`output`张量。

### 计算公式
该算子遵循标准的缩放点积注意力（Scaled Dot-Product Attention）计算框架。对于批次中的每一个序列，其计算公式如下：

$$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
-   $Q$: 查询（Query）张量，由输入 `query_nope` 和 `query_rope` 拼接而成。
-   $K$: 键（Key）张量，根据 `block_tables` 从 `kv_nope_cache` 和 `kv_rope_cache` 中动态重组得到。
-   $V$: 值（Value）张量，根据 `block_tables` 从 `kv_nope_cache` 中动态重组得到。
-   $d_k$: 键向量的维度，即 `head_size` 与 `head_size_rope` 之和。

### 计算过程与类型转换
算子内部的计算流程高度优化，以实现高效的内存访问和计算。

1.  **输入重组**：算子首先在逻辑上将 `query_nope` 和 `query_rope` 沿最后一个维度拼接，形成完整的查询张量 $Q$。同理，`kv_nope_cache` 和 `kv_rope_cache` 共同构成了完整的键缓存，而 `kv_nope_cache` 也被用作值缓存。
2.  **KV序列重构**：算子遍历批次中的每个序列。对于每个序列的历史Token（长度由 `k_seqlen_list` 指定），它使用 `block_tables` 将逻辑上的Token位置映射到KV缓存中的物理存储块（Block）和块内偏移（Offset），从而动态地重构出该序列所需的全量 $K$ 和 $V$ 张量。
3.  **分组查询注意力（GQA）**：在计算 $QK^T$ 时，算子会处理GQA逻辑。查询头的数量（`num_heads`）是键/值头数量（`kv_heads`）的整数倍。计算时，`num_heads` 个查询头被分成 `kv_heads` 组，每组内的查询头共享同一套 $K$ 和 $V$ 进行注意力计算。
4.  **数值精度**：为了在计算点积和Softmax时保持高精度并避免下溢或上溢，当输入数据类型为 `float16` 或 `bfloat16` 时，算子内部的累加过程和中间结果会提升至 `float32` 类型进行计算。
5.  **结果输出**：注意力计算的最终结果（`float32`类型）将被转换回输入张量的原始数据类型（`float16` 或 `bfloat16`），作为最终输出。

## 接口定义

### 算子原型定义接口
#### Input
- query_nope：Device侧的aclTensor。查询张量 $Q$ 中不包含旋转位置编码（RoPE）的部分。形状为 `(num_tokens, num_heads, head_size)`。数据类型支持float16、bfloat16，格式支持ND。
- query_rope：Device侧的aclTensor。查询张量 $Q$ 中仅包含RoPE的部分。形状为 `(num_tokens, num_heads, head_size_rope)`。数据类型支持float16、bfloat16，格式支持ND。
- kv_nope_cache：Device侧的aclTensor。分页缓存，存储所有序列的键张量 $K$ 的非RoPE部分以及完整的值张量 $V$。形状为 `(num_blocks, block_size, kv_heads, head_size)`。数据类型支持float16、bfloat16，格式支持ND。
- kv_rope_cache：Device侧的aclTensor。分页缓存，存储所有序列的键张量 $K$ 的RoPE部分。形状为 `(num_blocks, block_size, kv_heads, head_size_rope)`。数据类型支持float16、bfloat16，格式支持ND。
- block_tables：Device侧的aclTensor。一个二维整数张量，用于将逻辑Token索引映射到`kv_cache`中的物理块索引。形状为 `(batch_size, max_blocks_per_sequence)`。数据类型支持int32，格式支持ND。
#### Output
- out：Device侧的aclTensor。注意力计算的输出结果。形状为 `(num_tokens, num_heads, head_size)`。数据类型与输入保持一致，支持float16、bfloat16，格式支持ND。
#### Attr
- q_seqlen_list：一个整数列表，包含批处理中每个序列的查询长度。
- k_seqlen_list：一个整数列表，包含批处理中每个序列的键/值历史长度。

## 约束与限制
*   所有输入张量的数据类型必须一致，当前支持 `float16` 和 `bfloat16`。`block_tables` 的数据类型必须为 `int32`。
*   输入 `query_nope` 的 `num_tokens` 维度必须等于 `q_seqlen_list` 中所有长度的总和。
*   查询头的数量（`num_heads`）必须是键/值头数量（`kv_heads`）的整数倍。
*   `q_seqlen_list` 和 `k_seqlen_list` 列表的长度必须等于批处理大小（`batch_size`）。
*   `block_tables` 中存储的块索引必须是有效值，即小于 `kv_cache` 的 `num_blocks` 维度。