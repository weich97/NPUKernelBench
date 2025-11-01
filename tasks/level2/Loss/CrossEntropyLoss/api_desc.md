# aclnnCrossEntropyLoss

## 功能描述
### 算子功能
计算输入预测值（logits）与目标标签之间的交叉熵损失。该算子支持可选的权重、忽略索引、标签平滑和不同的归约（reduction）方式。此外，它还可以选择返回Z-loss相关值。

### 计算公式
**LogSoftmax:**
$$
\text{log\_softmax\_probs}_i = \text{predictions}_i - \left( \max_j(\text{predictions}_j) + \log \sum_j \exp(\text{predictions}_j - \max_j(\text{predictions}_j)) \right)
$$
或等价地，通过 LogSumExp ($LSE$):
$$
LSE = \max_j(\text{predictions}_j) + \log \sum_j \exp(\text{predictions}_j - \max_j(\text{predictions}_j))
$$
$$
\text{log\_softmax\_probs} = \text{predictions} - LSE
$$

**NLL Loss (Negative Log Likelihood Loss):**
对于每个样本 $i$:
$$
\text{nll\_loss\_i} = -\text{log\_softmax\_probs}_{i, \text{target}_i} \cdot \text{weight}_{\text{target}_i}
$$

**忽略索引 (ignore_index):**
如果 `ignore_index` >= 0，则对于 `target_labels_i == ignore_index` 的样本，其损失贡献为 0。

**标签平滑 (label_smoothing):**
$$
\text{loss}_{\text{smoothed}} = (1 - \text{label\_smoothing}) \cdot \text{base\_loss} + \text{label\_smoothing} \cdot \text{smooth\_term}
$$
其中，`smooth_term` 通常为：
$$
\text{smooth\_term} = -\frac{1}{C} \sum_j \text{log\_softmax\_probs}_j \cdot \text{weight}_j
$$
（这里的 $C$ 是类别数量）

**归约 (reduction):**
* `none`: 返回每个样本的损失 (N, )。
* `sum`: 返回所有有效样本损失的总和 (scalar)。
* `mean`: 返回所有有效样本损失的平均值 (scalar)，分母是有效权重之和。

**Z-loss (可选，如果 `returnZloss` 为 `True`):**
$$
\text{Zloss} = \text{lse\_square\_scale\_for\_zloss} \cdot \text{Mean}(LSE^2)
$$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.cross_entropy_loss()` 函数形式提供：

```python
def cross_entropy_loss(input, target, weight, reduction, ignore_index, label_smoothing, lse_square_scale_for_zloss, return_zloss):
    """
    计算输入预测值与目标标签之间的交叉熵损失。

    参数:
        input (Tensor): 预测值张量 (logits)，通常形状为 (N, C)，N 是批次大小，C 是类别数量。
        target (Tensor): 目标标签张量，形状为 (N,)，包含每个样本的类别索引。
        weight (Optional[Tensor]): 可选的类别权重张量，形状为 (C,)。如果提供，它会按类别加权损失。
        reduction (str): 损失的归约方式，可选 "none", "mean", "sum"。
        ignore_index (int): 指定一个在计算损失时应被忽略的目标值。通常为 -100。
        label_smoothing (double): 标签平滑的程度，值在 [0.0, 1.0] 之间。
        lse_square_scale_for_zloss (double): 用于 Z-loss 计算中 LogSumExp 平方项的缩放因子。
        return_zloss (bool): 如果为 True，则额外返回 Z-loss 和 LSE 值。

    返回:
        List[Tensor]: 包含四个张量的List：
                                      - 第一个张量 (`lossOut`) 是计算得到的交叉熵损失。
                                      - 第二个张量 (`logProbOut`) 是 LogSoftmax 后的概率张量。
                                      - 第三个张量 (`zlossOut`) 是 Z-loss 值（如果 `return_zloss` 为 False，可能为 0）。
                                      - 第四个张量 (`lseForZlossOut`) 是 LogSumExp 值。
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量和参数
batch_size = 4
num_classes = 10
input_dtype = torch.float32
target_dtype = torch.int64
weight_dtype = torch.float32

input_predictions = torch.randn([batch_size, num_classes], dtype=input_dtype)
target_labels = torch.randint(low=0, high=num_classes, size=(batch_size,), dtype=target_dtype)
weight = torch.randn(num_classes, dtype=weight_dtype)

ignore_index = -100
label_smoothing = 0.0
reduction = "mean"
lse_square_scale_for_zloss = 0.0
return_zloss = False

# 使用 cross_entropy_loss 执行操作
loss_out, log_prob_out, zloss_out, lse_for_zloss_out = kernel_gen_ops.cross_entropy_loss(
    input_predictions, target_labels, weight, reduction,
    ignore_index, label_smoothing, lse_square_scale_for_zloss, return_zloss
)

# Example with label smoothing and Z-loss
loss_out_smoothed, _, zloss_out_smoothed, _ = kernel_gen_ops.cross_entropy_loss(
    input_predictions, target_labels, None, "mean",
    -100, 0.1, 0.001, True # label_smoothing=0.1, return_zloss=True, lse_scale=0.001
)
```

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `input` 张量的维度必须是 **2维** (`(N, C)`)。
  * `target` 张量的维度必须是 **1维** (`(N,)`)，数值在[0, C)之间。
  * `input` 和 `logProbOut` 的形状、数据类型必须一致。
  * `target` 的数据类型支持 **INT32**、**INT64**。
  * `weightOptional` 如果提供，其形状必须是 **1维** (`(C,)`)，且数据类型通常与 `input` 一致。
  * `lossOut` 的形状取决于 `reduction`：
    * `"none"`: 形状与 `target` 相同 (`(N,)`)。
    * `"mean"` / `"sum"`: 形状为标量 `()`。
  * `zlossOut` 的形状为标量 `()`。
  * `lseForZlossOut` 的形状为 `(N,)`。
  * `input`、`weightOptional` 的数据类型支持 **FLOAT16**、**FLOAT**、**BFLOAT16**。
  * `ignoreIndex` 必须为 **INT64**。
  * `labelSmoothing`、`lseSquareScaleForZloss` 必须为**浮点数**。
  * `reduction` 必须是字符串 `"none"`、`"mean"` 或 `"sum"` 之一。