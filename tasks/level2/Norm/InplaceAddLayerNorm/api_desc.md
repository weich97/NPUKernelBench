# aclnnInplaceAddLayerNorm

## 功能描述
### 算子功能
对两个输入张量执行逐元素相加后进行层归一化（LayerNorm）操作，并通过内存复用机制显著减少内存占用。

### 计算公式
  $$x2 = x1 + x2_{input} + bias$$
  $$x1 = \frac{x2 - \overline{x2}}{\sqrt{Var(x2) + \epsilon}} * \gamma + \beta$$

## 接口定义

* **Inputs (输入)**
    * `x1` (aclTensor\*, 计算输入): `REQUIRED`. 表示AddLayerNorm中加法计算的输入，将会在算子内做 `x1 + x2 + bias` 的计算并对计算结果做层归一化。此输入为in-place输入，计算结果将直接写回该Tensor。
        * 支持设备: 昇腾910B AI处理器, 昇腾910_93 AI处理器。
        * 数据类型: 支持 `FLOAT16`, `BFLOAT16`, `FLOAT`。
        * 数据格式: 支持 `ND`。
    * `x2` (aclTensor\*, 计算输入): `REQUIRED`. 表示AddLayerNorm中加法计算的输入。
        * 支持设备: 昇腾910B AI处理器, 昇腾910_93 AI处理器。
        * 数据类型: 支持 `FLOAT16`, `BFLOAT16`, `FLOAT`。
        * 数据格式: 支持 `ND`。
    * `gamma` (aclTensor\*, 计算输入): `REQUIRED`. 对应LayerNorm计算公式中的 gamma 参数。
        * 支持设备: 昇腾910B AI处理器, 昇腾910_93 AI处理器。
        * 数据类型: 支持 `FLOAT16`, `BFLOAT16`, `FLOAT`。
        * 数据格式: 支持 `ND`。
    * `beta` (aclTensor\*, 计算输入): `REQUIRED`. 对应LayerNorm计算公式中的 beta 参数。
        * 支持设备: 昇腾910B AI处理器, 昇腾910_93 AI处理器。
        * 数据类型: 支持 `FLOAT16`, `BFLOAT16`, `FLOAT`。
        * 数据格式: 支持 `ND`。
    * `bias` (aclTensor\*, 计算输入): `OPTIONAL`. 表示AddLayerNorm中加法计算的可选偏置项。
        * 支持设备: 昇腾910B AI处理器, 昇腾910_93 AI处理器。
        * shape可以和gamma/beta（broadcast）或是和x1/x2一致（present）
        * 数据类型: 支持 `FLOAT16`, `BFLOAT16`, `FLOAT`。
        * 数据格式: 支持 `ND`。

* **Outputs (输出)**
    * `x1` (aclTensor\*, 计算输出): `REQUIRED`. in-place输出，即LayerNorm的最终计算结果。shape及数据类型与输入`x1`一致。
    * `mean` (aclTensor\*, 计算输出): `REQUIRED`. LayerNorm计算过程中产生的均值。
        * 数据类型: `FLOAT`。
        * 数据格式: 支持 `ND`。
    * `rstd` (aclTensor\*, 计算输出): `REQUIRED`. LayerNorm计算过程中产生的方差的平方根的倒数。
        * 数据类型: `FLOAT`。
        * 数据格式: 支持 `ND`。
    * `x2` (aclTensor\*, 计算输出): `REQUIRED`. 当属性 `additional_output` 设置为 `true` 时，此输出为 `x1 + x2 + bias` 的中间结果。
        * 数据类型: 支持 `FLOAT16`, `BFLOAT16`, `FLOAT`。
        * 数据格式: 支持 `ND`。

* **Attributes (属性)**
    * `epsilon`: `OPTIONAL`. LayerNorm公式中为保证数值稳定性加在分母上的值。
        * 属性类型: `Float`。
        * 默认值: `1e-5`。
    * `additional_output`: `OPTIONAL`. bool类型的属性，表示是否将 `x1 + x2 + bias` 的中间结果额外输出到output `x2`。
        * 属性类型: `Bool`。
        * 默认值: `false`。


## 约束与限制
- **功能维度**
  * 数据格式支持：ND。
  * eps默认只支持1e-5。
  * 张量形状维度不高于8维，数据格式支持ND。
  * 输出张量 y 与输入张量 x1、x2 具有相同的形状、数据类型和数据格式。
  * gamma、beta 的形状必须与归一化维度一致。
