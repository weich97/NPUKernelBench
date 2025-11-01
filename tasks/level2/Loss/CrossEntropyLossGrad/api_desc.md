# aclnnCrossEntropyLossGrad

## 功能描述
### 算子功能
计算交叉熵损失函数相对于其输入（通常是 LogSoftmax 的输出）的梯度。此算子用于反向传播过程。

### 计算公式
交叉熵损失函数的梯度计算比较复杂，涉及 LogSoftmax 及其损失的链式法则。
基本梯度：
$$
\frac{\partial L}{\partial \mathbf{x}} = \mathbf{p} - \mathbf{y}
$$
其中，$\mathbf{L}$ 是交叉熵损失，$ \mathbf{x} $ 是 Logits (在 LogSoftmax 之前)，$ \mathbf{p} $ 是预测的概率分布 (Softmax 的输出)，$ \mathbf{y} $ 是真实标签的独热编码（One-Hot Encoding）。
对于带有 LogSoftmax 的交叉熵损失，梯度通常直接从 LogSoftmax 输出 (即 `logProb`) 传播：
$$
\frac{\partial L}{\partial \text{logProb}_i} = \text{softmax}(\mathbf{x})_i - \text{one\_hot\_target}_i
$$
在实际的算子实现中，还会考虑 `reduction` (求和、平均、不处理)、`weight` (类别权重)、`ignore_index` (忽略的标签索引) 和 `label_smoothing` (标签平滑) 等因素。如果启用了 ZLoss，还会包含 ZLoss 项的梯度贡献。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.cross_entropy_loss_grad()` 函数形式提供：

```python
def cross_entropy_loss_grad(
    grad_loss: torch.Tensor,
    log_prob: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor],
    grad_zloss: Optional[torch.Tensor],
    lse_for_zloss: Optional[torch.Tensor],
    reduction: str,
    ignore_index: int,
    label_smoothing: float,
    lse_square_scale_for_zloss: float
) -> List[torch.Tensor]:
    """
    计算交叉熵损失函数相对于其输入（LogSoftmax的输出）的梯度。

    参数:
        grad_loss (Tensor): 从下游层传入的损失输出的梯度。
                            如果reduction为"none"，形状为 (N,)；否则为标量。
        log_prob (Tensor): LogSoftmax层的输出，形状为 (N, C)。
        target (Tensor): 真实标签，形状为 (N,)。
        weight (Optional[Tensor]): 类别权重，形状为 (C,)。
        grad_zloss (Optional[Tensor]): ZLoss的梯度，如果启用ZLoss则传入，通常为标量。
        lse_for_zloss (Optional[Tensor]): 用于ZLoss计算的LogSumExp值，形状为 (N,)。
        reduction (str): 损失的归约方式，可选值："none"、"mean"、"sum"。
        ignore_index (int): 在计算损失时要忽略的标签索引。
        label_smoothing (float): 标签平滑的因子。
        lse_square_scale_for_zloss (float): 用于ZLoss计算的平方LSE的缩放因子。

    返回:
        List[Tensor]: 包含一个张量的列表，即相对于原始输入（通常是Logits）的梯度 `grad_input`。
                      `grad_input` 的形状与 `log_prob` 相同。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 假设来自前向传播的输出
batch_size = 4
num_classes = 10
input_dtype = torch.float32

# 1. 模拟前向传播的输出 (通常是LogSoftmax的结果)
# 模拟 logits
logits = torch.randn(batch_size, num_classes, dtype=input_dtype)
# 模拟 log_prob (通常是 log_softmax(logits))
log_prob = torch.log_softmax(logits, dim=-1)

# 模拟真实标签
target = torch.randint(0, num_classes, (batch_size,), dtype=torch.int64)

# 模拟损失的梯度 (假设损失是一个标量，并且其梯度是1.0)
# 如果前向传播的reduction是"none"，grad_loss应是 (batch_size,)
grad_loss = torch.tensor(1.0, dtype=input_dtype) 

# 其他参数
weight = torch.rand(num_classes, dtype=input_dtype) # 类别权重
ignore_index = -100
label_smoothing = 0.0
reduction = "mean"
lse_square_scale_for_zloss = 0.001

# 假设ZLoss相关参数，如果前向传播启用了ZLoss
grad_zloss = torch.tensor(0.5, dtype=input_dtype) # 假设ZLoss的梯度
# 模拟前向传播中计算的 lse (LogSumExp)
lse_for_zloss = torch.randn(batch_size, dtype=input_dtype) 

# 调用 CrossEntropyLossGrad
grad_input = kernel_gen_ops.cross_entropy_loss_grad(
    grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss,
    reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss
)[0] # 返回值是列表，取第一个元素

# 如果不使用ZLoss相关的梯度，则传入None
grad_input_no_zloss = kernel_gen_ops.cross_entropy_loss_grad(
    grad_loss, log_prob, target, weight, None, None,
    reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss
)[0]
```

### 约束与限制
## 约束与限制
- **功能维度**
  * 数据格式支持：ND。
  * `grad_loss`、`log_prob`、`weight`、`grad_zloss`、`lse_for_zloss` 必须是浮点类型 (float16, float32, bfloat16)。
  * `target` 必须是整型 (int32, int64)。
  * `log_prob` 的形状通常为 `(N, C)`，其中 `N` 是批次大小，`C` 是类别数。
  * `target` 的形状通常为 `(N,)`。
  * `weight` 的形状必须为 `(C,)`。
  * `grad_loss` 的形状取决于 `reduction`：如果 `reduction` 为 "none"，则形状为 `(N,)`；否则为标量。
  * `grad_zloss` 必须是标量。
  * `lse_for_zloss` 的形状通常为 `(N,)`。
  * `ignore_index` 的值 `-100` 通常表示不忽略任何标签。
  * `lse_square_scale_for_zloss` 仅在 `grad_zloss` 和 `lse_for_zloss` 存在时生效。
  * 输出 `grad_input` 的形状与 `log_prob` 相同，数据类型与 `log_prob` 相同。