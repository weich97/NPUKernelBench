# GeluQuant

## 功能
一步完成 **GELU 激活** + **量化**（支持静态 / 动态两种模式），输出 8-bit 整型张量及反量化比例因子，仅考虑Per-tensor场景。

## 公式
设输入张量 `x`，可选参数 `approximate` 为激活模式，其余符号说明见下：

1. **GELU 激活**
   $$
   \text{gelu} = \text{GELU}(x,\ \text{approximate})
   $$

2. **静态量化**（`quant_mode == "static"`）
   $$
   y_{\text{out}} = \Bigl\lfloor \text{gelu} \cdot \text{inputScale} + \text{inputOffset} \Bigr\rceil
   \ \text{clip}_{[-128,\,127]}
   $$
   - `outScaleOut` 不更新，保持全 0。

3. **动态量化**（`quant_mode == "dynamic"`）
   $$
   \begin{aligned}
   \text{temp} &= \text{gelu} \cdot \text{inputScale} \\
   \text{maxAbs} &= \max(|\text{temp}|) \\[2pt]
   \text{outScaleOut} &= \frac{\text{maxAbs}}{127.0} \\[4pt]
   y_{\text{out}} &= \Bigl\lfloor \frac{\text{temp}}{\text{outScaleOut}} \Bigr\rceil
   \ \text{clip}_{[-128,\,127]}
   \end{aligned}
   $$

## 支持的数据类型与格式

| 张量                | 数据类型                              | 数据格式 | 维度约束 |
|---------------------|---------------------------------------|----------|----------|
| x                   | `float16`, `float32`, `bfloat16`      | ND       | 2–8 维   |
| inputScaleOptional  | 与 x 同 dtype                         | ND       | 1 维     |
| inputOffsetOptional | 与 inputScaleOptional 同 dtype & shape| ND       | 1 维     |
| yOut                | `int8`                                | ND       | 与 x 同  |
| outScaleOut         | `float32`                             | ND       | 与 x 同，仅最后一位维度为 1 |

> 当 `quant_mode == "static"` 时，`inputScaleOptional` 为必选，`inputOffsetOptional` 可选；  
> 当 `quant_mode == "dynamic"` 时，两者均可省略。

## Python 接口
```python
def gelu_quant(
    x: torch.Tensor,
    input_scale: Optional[torch.Tensor] = None,
    input_offset: Optional[torch.Tensor] = None,
    approximate: str = "tanh",
    quant_mode: str = "static"
) -> List[torch.Tensor]:
    """
    GELU + 量化

    参数
    ----
    x : torch.Tensor
        Device 侧输入张量；dtype ∈ {float16, float32, bfloat16}；shape 2–8 维。
    input_scale : torch.Tensor, optional
        缩放系数；dtype 与 x 相同；1-D，长度 = x.shape[-1] 或 1。
    input_offset : torch.Tensor, optional
        零点偏移；dtype/shape 与 input_scale 一致。
    approximate : {"tanh", "none"}, default "tanh"
        GELU 近似方式。
    quant_mode : {"static", "dynamic"}, default "static"
        量化模式。

    返回
    ----
    y_out : torch.Tensor
        量化后的 int8 张量，shape 与 x 相同。
    out_scale : torch.Tensor
        反量化比例因子 (float32)。  
        - 静态模式下为全 0；  
        - 动态模式下为计算得到的 per-token 比例。
    """
import torch
import kernel_gen_ops

x = torch.randn(4, 8, dtype=torch.float16, device='npu')

# 静态量化
scale = torch.full((8,), 0.1, dtype=torch.float16, device='npu')
offset = torch.zeros(8, dtype=torch.float16, device='npu')
y_static, scale_static = kernel_gen_ops.gelu_quant(
    x, scale, offset, approximate="tanh", quant_mode="static"
)

# 动态量化
y_dynamic, scale_dynamic = kernel_gen_ops.gelu_quant(
    x, quant_mode="dynamic"
)

## 约束与限制

x、inputScaleOptional、inputOffsetOptional的数据类型只支持FLOAT16，FLOAT32, BF16，数据格式只支持ND。