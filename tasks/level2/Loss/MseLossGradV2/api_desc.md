# aclnnMseLossGrad

## 功能描述
### 算子功能
计算均方误差损失（MSE Loss）的反向传播梯度。它根据损失的梯度（`gradOutput`）、预测值（`self`）和目标值（`target`）来计算对预测值的梯度。

### 计算公式
假设 MSE Loss 前向计算为：
$$
\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} (x_i - y_i)^2 \quad (\text{for reduction='mean'})
$$
$$
\text{Loss} = \sum_{i=1}^{N} (x_i - y_i)^2 \quad (\text{for reduction='sum'} \text{ or 'none'})
$$
其中 $x$ 是预测值（`self`），$y$ 是目标值（`target`）。

反向传播的梯度（`out`）计算公式为：
$$
\text{out}_i = \text{gradOutput}_i \cdot \text{coefficient} \cdot (x_i - y_i)
$$
其中，`coefficient` 取决于 `reduction` 类型：
* 如果 `reduction` 为 `"mean"`：`coefficient` 为 $\frac{2}{N_{total}}$，其中 $N_{total}$ 是 `x` 张量中所有元素的总数。
* 如果 `reduction` 为 `"sum"` 或 `"none"`：`coefficient` 为 $2$。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.mse_loss_grad()` 函数形式提供：

```python
def mse_loss_grad(gradOutput, self_input, target_input, reduction):
    """
    计算均方误差损失的反向传播梯度。

    参数:
        gradOutput (Tensor): 损失函数输出的梯度张量。
        self_input (Tensor): 预测值张量（MSE Loss 前向的输入 `x`）。
        target_input (Tensor): 目标值张量（MSE Loss 前向的输入 `y`）。
        reduction (str): 前向损失的归约方式，可选 "none", "mean", "sum"。

    返回:
        Tensor: 对预测值 `self_input` 的梯度张量。
    """
    import torch
    import kernel_gen_ops

    # 创建输入张量和参数
    input_shape = [2, 2]
    dtype = torch.float32
    reduction = "mean"

    input_predict = torch.randn(input_shape, dtype=dtype)
    input_label = torch.randn(input_shape, dtype=dtype)
    input_dout = torch.randn(input_shape, dtype=dtype) # Gradient from downstream

    # 使用 mse_loss_grad 计算反向梯度
    out_grad = kernel_gen_ops.mse_loss_grad(input_dout, input_predict, input_label, reduction)

    print("Output gradient shape:", out_grad.shape)

    # Example with 'sum' reduction
    out_grad_sum = kernel_gen_ops.mse_loss_grad(
        torch.randn(input_shape, dtype=dtype),
        torch.randn(input_shape, dtype=dtype),
        torch.randn(input_shape, dtype=dtype),
        "sum"
    )
    print("Output gradient shape (sum reduction):", out_grad_sum.shape)

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * 'gradOutput`、`self`（预测值）、`target`（目标值）和 `out` 的形状和数据类型必须一致。
  * 'gradOutput`、`self`、`target` 的数据类型支持 **FLOAT16**、**FLOAT**、**BFLOAT16**、**INT32**、**INT64**、**DOUBLE**、**INT8** (实际支持取决于 NPU 硬件)。
  * 'reduction` 必须是字符串 `"none"`、`"mean"` 或 `"sum"` 之一。
  * 'reduction` 参数的映射：`"none"` 对应 `0`，`"mean"` 对应 `1`，`"sum"` 对应 `2`。