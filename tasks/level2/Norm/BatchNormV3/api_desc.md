# aclnnBatchNormV3

## 功能描述
### 算子功能
Batch Normalization (BatchNorm) 是一种常用的神经网络层，用于规范化迷你批次（mini-batch）的输入。它对输入在通道维度上进行归一化，并可选地使用可学习的缩放（`weight`）和偏移（`bias`）参数。在训练模式下，它还会更新运行中的均值和方差统计量。

### 计算公式
**训练模式 (training=True) 下的归一化：**
$$
\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i
$$
$$
\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2
$$
$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
$$
$$
y_i = \gamma \hat{x}_i + \beta
$$

**训练模式下运行均值和方差的更新：**
$$
running\_mean = (1 - \text{momentum}) \cdot running\_mean + \text{momentum} \cdot \mu_B
$$
$$
running\_var = (1 - \text{momentum}) \cdot running\_var + \text{momentum} \cdot \sigma_B^2
$$

**评估模式 (training=False) 下的归一化：**
$$
\hat{x}_i = \frac{x_i - running\_mean}{\sqrt{running\_var + \epsilon}}
$$
$$
y_i = \gamma \hat{x}_i + \beta
$$

其中：
* $x_i$ 是当前批次输入的一个元素。
* $m$ 是当前批次中元素的数量。
* $\mu_B$ 和 $\sigma_B^2$ 是当前批次的均值和方差。
* $running\_mean$ 和 $running\_var$ 是模型在训练过程中积累的全局统计量。
* $\gamma$ (weight) 和 $\beta$ (bias) 是可学习的缩放和偏移参数。
* $\epsilon$ 是一个添加到方差上的小常数，用于数值稳定性。
* $momentum$ 是用于更新 $running\_mean$ 和 $running\_var$ 的参数。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.batch_norm_v3()` 函数形式提供：

```python
def batch_norm_v3(input_tensor, weight, bias, running_mean, running_var, training, momentum, eps):
    """
    对输入张量执行 Batch Normalization V3 操作。

    参数:
        input_tensor (Tensor): 输入Device侧张量。
        weight (Optional[Tensor]): 可选的缩放参数张量，形状与特征维度相同。如果为 None，则不进行缩放。
        bias (Optional[Tensor]): 可选的偏移参数张量，形状与特征维度相同。如果为 None，则不进行偏移。
        running_mean (Tensor): 运行均值张量，在训练模式下会被更新。
        running_var (Tensor): 运行方差张量，在训练模式下会被更新。
        training (bool): 布尔值，指示当前是训练模式 (True) 还是评估模式 (False)。
        momentum (double): 用于更新运行均值和方差的动量参数。
        eps (double): 用于数值稳定性的一个小常数。

    返回:
        List[Tensor]: 包含三个张量的list：
                                      - 第一个张量是归一化后的输出张量 (`output`)。
                                      - 第二个张量是当前批次的均值 (`saveMean`)。
                                      - 第三个张量是当前批次标准差的倒数 (`saveInvstd`)。
    """

```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量和参数
input_shape = [1, 2, 4]
num_features = 2 # Number of features (channels)
dtype = torch.float32
eps = 1e-5
momentum = 0.1
training = True
affine = True # If weight and bias are used

input_tensor = torch.randn(input_shape, dtype=dtype)

weight = torch.randn(num_features, dtype=dtype) if affine else None
bias = torch.randn(num_features, dtype=dtype) if affine else None

# Running statistics (will be updated in-place by the operator if training=True)
running_mean = torch.zeros(num_features, dtype=dtype)
running_var = torch.ones(num_features, dtype=dtype)

# 使用 batch_norm_v3 执行操作
output, save_mean, save_invstd = kernel_gen_ops.batch_norm_v3(
    input_tensor, weight, bias, running_mean, running_var, training, momentum, eps
)

# Example for inference mode
input_inference = torch.randn(input_shape, dtype=dtype)
# Use previously updated running_mean and running_var
output_inference, _, _ = kernel_gen_ops.batch_norm_v3(
    input_inference, weight, bias, running_mean, running_var, False, momentum, eps # training=False
)

```

## 约束与限制
- **功能维度**
  * 数据格式支持：**ND**。
  * `input` 和 `output` 的形状和数据类型必须一致。
  * `weight` 和 `bias` 是可选的，如果提供，它们的形状必须与**特征维度**（通道数）匹配。
  * `runningMean` 和 `runningVar` 必须是 **1D 张量**，形状与特征维度匹配，且数据类型与 `input` 一致。在训练模式下，它们会被**原地更新**。
  * `input`、`weight`、`bias`、`runningMean`、`runningVar` 的数据类型支持 **FLOAT16**、**FLOAT**、**BFLOAT16**。
  * `momentum` 必须为**浮点数**，范围通常在 [0, 1] 之间。
  * `eps` 必须为**浮点数**，double类型，且通常取值如 1e-5 或 1e-6。
  * 算子会返回当前批次的 'mean' 和 'invstd'，即使在评估模式下这些值不用于计算输出，它们也可能作为中间输出返回。