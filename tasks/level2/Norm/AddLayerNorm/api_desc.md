# aclnnAddLayerNorm

## 功能描述
### 算子功能
对两个输入张量执行逐元素相加后进行层归一化（LayerNorm）操作。

### 计算公式
  $$
  x = x1 + x2 + bias
  $$

  $$
  y = {{x-\bar{x}}\over\sqrt {Var(x)+eps}} * \gamma + \beta
  $$

## 接口定义
  * x1（aclTensor \*，计算输入）：表示AddLayerNorm中加法计算的输入，将会在算子内做 x1 + x2 + bias 的计算并对计算结果做层归一化；是Device 侧的aclTensor，shape支持1-8维度，不支持输入的某一维的值为0，数据格式支持ND。
    * 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * x2（aclTensor \*，计算输入）：表示AddLayerNorm中加法计算的输入，将会在算子内做 x1 + x2 + bias 的计算并对计算结果做层归一化；是Device 侧的aclTensor，shape需要与x1一致，数据格式支持ND。
    * 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * beta（aclTensor \*，计算输入）：对应LayerNorm计算公式中的 beta ，表示层归一化中的 beta 参数；是Device 侧的aclTensor，shape支持1-8维度，与x1需要norm的维度的维度值相同，数据格式支持ND。
    * 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * gamma（aclTensor \*，计算输入）：对应LayerNorm计算公式中的 gamma，表示层归一化中的 gamma 参数；是Device 侧的aclTensor，shape支持1-8维度，与x1需要norm的维度的维度值相同，数据格式支持ND。
    * 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * bias（aclTensor \*，计算输入）：可选输入参数，表示AddLayerNorm中加法计算的输入，将会在算子内做 x1 + x2 + bias 的计算并对计算结果做层归一化；shape可以和gamma/beta（broadcast）或是和x1/x2一致（present），是Device 侧的aclTensor，shape支持1-8维度，数据格式支持ND。
    * 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * epsilon（double \*，计算输入）：公式中的输入`eps`，添加到分母中的值，以确保数值稳定；host侧的aclScalar，数据类型为double，仅支持取值1e-5。
  * additionalOut（bool \*，计算输入）：表示是否开启x=x1+x2+bias的输出，host侧的aclScalar，数据类型为bool。
  * meanOut（aclTensor \*，计算输出）：输出 LayerNorm 计算过程中 (x1 + x2 + bias) 的结果的均值，Device 侧的aclTensor，数据类型为FLOAT32，shape需要与x1满足broadcast关系（前几维的维度和x1前几维的维度相同，前几维指x1的维度减去gamma的维度，表示不需要norm的维度），数据格式支持ND。计算逻辑：mean = np.mean(x1 + x2 + bias)。
  * rstdOut（aclTensor \*，计算输出）：输出 LayerNorm 计算过程中 rstd 的结果，Device 侧的aclTensor，数据类型为FLOAT32，shape需要与x1满足broadcast关系（前几维的维度和x1前几维的维度相同），数据格式支持ND。计算逻辑：rstd = np.power((np.var(x1 + x2 + bias) + epsilon), (-0.5))。
  * yOut（aclTensor \*，计算输出）：表示LayerNorm的结果输出y，Device 侧的aclTensor，shape需要与输入x1/x2一致，数据格式支持ND。
    * 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。
  * xOut（aclTensor \*，计算输出）：表示LayerNorm的结果输出x，Device 侧的aclTensor，shape需要与输入x1/x2一致，数据格式支持ND。
    * 昇腾910B AI处理器、昇腾910_93 AI处理器：数据类型支持FLOAT32、FLOAT16、BFLOAT16。

## 约束与限制
- **功能维度**
  * 数据格式支持：ND。
  * eps默认只支持1e-5。
  * 张量形状维度不高于8维，数据格式支持ND。
  * 输出张量 y 与输入张量 x1、x2 具有相同的形状、数据类型和数据格式。
  * gamma、beta 的形状必须与归一化维度一致。
