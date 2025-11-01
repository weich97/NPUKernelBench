# aclnnForeachNonFiniteCheckAndUnscale

## 功能描述
---
### 算子功能
`aclnnForeachNonFiniteCheckAndUnscale` 操作用于在混合精度训练的反向传播过程中，对一系列缩放后的梯度张量进行非有限值（NaN 或 Inf）检查，并同时执行梯度反缩放操作。如果发现任何非有限值，一个 **浮点标量** 将被设置为 `1.0`；否则保持 `0.0`。

### 计算公式
---

对于输入缩放后的梯度张量列表 `scaledGrads` ($X = [X_0, X_1, ..., X_{n-1}]$) 和一个缩放因子 `inScale` ($S$)：

1.  **非有限值检查**: 对于列表中的每个张量 $X_i$，检查其中是否存在任何元素 $x$ 满足 $x = \text{NaN}$ 或 $x = \pm \infty$。
    * 如果发现任何非有限值，则将浮点标志 `foundInf` 设置为 `1.0`。
    * 否则，`foundInf` 保持为 `0.0`。

2.  **梯度反缩放 (In-Place)**: 对于列表中的每个张量 $X_i$，计算其反缩放后的梯度并覆盖原 $X_i$：
    $$X_i \leftarrow \frac{X_i}{S}$$

**输出**:
* `scaledGrads` (修改后)：现在包含反缩放后的梯度张量列表。
* `foundInf` (修改后)：一个标量 **浮点** 张量，取值为 `1.0`（发现非有限值）或 `0.0`（未发现）。

## 接口定义
---
### Python 接口
通过 PyBind11 封装，在 Python 中以 `kernel_gen_ops.foreach_non_finite_check_and_unscale()` 提供：

```python
def foreach_non_finite_check_and_unscale(
    scaled_grads: List[torch.Tensor],
    found_inf: torch.Tensor,
    in_scale: torch.Tensor
) -> Tuple[List[torch.Tensor], torch.Tensor]:
    """
    对缩放后的梯度列表进行非有限值检查和反缩放（in-place）。

    参数:
        scaled_grads (List[torch.Tensor]): 输入 Device 侧缩放后的梯度张量列表，**原地**反缩放回写。
        found_inf   (torch.Tensor):        标量 **浮点** 张量（dtype=float16/float32/bfloat16）。  
                                          调用前只需分配空间；函数写入 1.0（发现非有限值）或 0.0（未发现）。
        in_scale    (torch.Tensor):        标量张量，**浮点类型**，反缩放系数。

    返回:
        Tuple[List[torch.Tensor], torch.Tensor]:
            - scaled_grads: 已原地反缩放的梯度列表。
            - found_inf:    已写入 1.0/0.0 的标量浮点张量。

    注意:
        - `scaled_grads` 所有张量须同 dtype，且为 float16/float32/bfloat16。
        - shape 维度 ≤ 8，数据格式支持 ND，支持非连续。
        - `found_inf`、`in_scale` 必须是标量，且 dtype 与 `scaled_grads` 保持一致。
        - `in_scale` 建议 > 0，确保数值稳定性。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 示例 1: 不含非有限值
scaled_grads_list_clean = [
    torch.randn(2, 2, dtype=torch.float32) * 1000, # 假设是放大1000倍的梯度
    torch.randn(3, 3, dtype=torch.float32) * 1000
]
in_scale_clean = torch.tensor(1000.0, dtype=torch.float32)
found_inf_check_clean = torch.tensor(0, dtype=torch.int8) # 预分配输出标志张量

unscaled_grads_clean, found_inf_clean = kernel_gen_ops.foreach_non_finite_check_and_unscale(
    scaled_grads_list_clean,
    found_inf_check_clean,
    in_scale_clean
)

# 示例 2: 包含非有限值 (NaN)
scaled_grads_list_with_nan = [
    torch.randn(2, 2, dtype=torch.float32) * 1000,
    torch.randn(3, 3, dtype=torch.float32) * 1000
]
# 注入 NaN
scaled_grads_list_with_nan[0].flatten()[0] = float('nan')
in_scale_nan = torch.tensor(1000.0, dtype=torch.float32)
found_inf_check_nan = torch.tensor(0, dtype=torch.int8) # 预分配输出标志张量

unscaled_grads_nan, found_inf_nan = kernel_gen_ops.foreach_non_finite_check_and_unscale(
    scaled_grads_list_with_nan,
    found_inf_check_nan,
    in_scale_nan
)


# 示例 3: 包含非有限值 (Inf)
scaled_grads_list_with_inf = [
    torch.randn(2, 2, dtype=torch.float16) * 1000, # Using float16 for easier Inf generation
    torch.randn(3, 3, dtype=torch.float16) * 1000
]
# 注入 Inf
scaled_grads_list_with_inf[1].flatten()[1] = float('inf')
in_scale_inf = torch.tensor(512.0, dtype=torch.float16)
found_inf_check_inf = torch.tensor(0, dtype=torch.int8)

unscaled_grads_inf, found_inf_inf = kernel_gen_ops.foreach_non_finite_check_and_unscale(
    scaled_grads_list_with_inf,
    found_inf_check_inf,
    in_scale_inf
)

```

## 约束与限制
---
- **数据格式**: 支持 ND 格式。
- **数据类型**：`scaled_grads` 列表中的所有张量以及 `in_scale` 必须是**浮点类型** (float16, float32, bfloat16)。
- **`found_inf_out` 类型**：`found_inf_out` 必须是**标量**。
- **张量维度**：所有输入张量的 **shape 维度不高于8维**。
- **数据连续性**：支持**非连续的 Tensor**。
- **输出特性**：
    - 输出 `unscaled_grads` 列表中的张量与输入 `scaled_grads` 中的张量具有相同的**形状**、**数据类型**和**数据格式**。
    - `found_inf_out` 和 `in_scale` 必须是**标量张量**。
    - `in_scale` 应为**正数**，以确保反缩放操作的数值稳定性。