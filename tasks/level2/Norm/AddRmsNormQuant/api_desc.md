# aclnnAddRmsNormQuant

## 功能描述
### 算子功能
是大模型常用的标准化操作，相比LayerNorm算子，其去掉了减去均值的部分。AddRmsNormQuant算子将RmsNorm前的Add算子以及RmsNorm后的Quantize算子融合起来，减少搬入搬出操作。



## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.add_rms_norm_quant()` 函数形式提供：

```python
def add_rms_norm_quant(x1, x2, gamma, scales1, zero_points1, scales2=None, zero_points2=None, axis=-1, epsilon=1e-6, div_mode=True):
    """
    对两个输入张量执行逐元素相加后进行 RMS 归一化（RMSNorm）操作，并进行量化处理。

    参数:
        x1 (Tensor): 第一个输入 Device 侧张量。类型通常为 bfloat16 或 float16。
        x2 (Tensor): 第二个输入 Device 侧张量，与 x1 具有相同的形状和数据类型。
        gamma (Tensor): 缩放参数张量，用于 RMSNorm。形状与需要归一化的维度一致。
        scales1 (Tensor): 第一路输出的量化 scale，float32 类型，用于归一化后量化 y1。
        zero_points1 (Tensor): 第一路输出的量化 zero point，float32 类型，用于归一化后量化 y1。
        scales2 (Tensor, 可选): 第二路输出的量化 scale。若为 None，则 y2 返回零张量。
        zero_points2 (Tensor, 可选): 第二路输出的量化 zero point。若为 None，则 y2 返回零张量。
        axis (int): 进行归一化的维度索引，默认为最后一个维度（-1）。
        epsilon (float): 防止除以 0 的小常数，默认值为 1e-6。
        div_mode (bool): 是否以除法方式进行归一化（True 表示除以 RMS，False 表示乘以 rsqrt）。

    返回:
        List[Tensor]: 返回包含两个张量的列表：
            - y1 (Tensor): 执行 RMSNorm 并量化后的主输出，类型为 int8，形状与 x1 相同。
            - y2 (Tensor): 备用输出，若提供 scales2 和 zero_points2 则执行量化，否则返回与 x1 同形状的 int8 零张量。
    """


```
## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量和参数
x1 = torch.rand((4, 2048), dtype=torch.float16)
x1 = torch.rand((4, 2048), dtype=torch.float16)

gamma = torch.rand(2048, dtype=torch.float16)

scales1 = torch.rand(2048, dtype=torch.float32)
scales2 = None # c++底层实现默认为空
zero_points1 = torch.rand(shape, dtype=torch.int32)
zero_points2 = None # c++底层实现默认为空

axis = -1
epsilon = 1e-5
div_mode = True

# 使用 add_layer_norm 对两个输入张量执行逐元素相加后进行层归一化（LayerNorm）操作
y = kernel_gen_ops.add_rms_norm_quant(x1, x2, gamma, scales1, scales2, zero_points1, zero_points2, axis, epsilon, div_mode)
```


## 约束与限制
- **功能维度**
  * 数据格式支持：ND。
  * 张量形状维度不高于8维，数据格式支持ND。
  * 输出张量 y 与输入张量 x1、x2 具有相同的形状、数据类型和数据格式。
