## Kernel Generator

NPU Kernel Bench 中的 Kernel Generator 模块利用大语言模型（LLM）辅助生成昇腾平台上的算子代码。其核心范式依赖于精心构造的 Prompt 和明确的生成目标，旨在让 LLM 理解算子需求并输出结构化的代码文件。

### 1. Prompt的构成

向大语言模型提供的 Prompt 是指导其生成代码的关键。它通常由以下几个部分动态组合而成：

#### a. 任务定义与API描述 (`api_desc.md`)

这是 Prompt 的核心输入。每个算子任务目录下都有一个 `api_desc.md` 文件，它详细描述了算子的规格和要求。主要包含：

* **算子功能描述：** 解释算子需要完成的数学运算或逻辑功能。
* **输入输出张量信息：**
    * 名称、数据类型（如 `float16`, `float32`, `int32` 等）。
    * 形状（Shape），可能包含动态维度。
    * 布局（Layout/Format，如 `ND`, `NCHW` 等）。
* **属性（Attributes）：** 算子特有的参数及其描述和类型。
* **约束条件与注意事项：** 例如数据对齐要求、取值范围、特定硬件行为等。
* **数学公式 (可选)：** 清晰的数学表达式。

**示例片段 (`api_desc.md`)**:
```markdown
# AddCustom

## 功能描述

### 算子功能
实现了两个数据相加，返回相加结果的功能。

### 计算公式

  $$
  z = x + y
  $$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.add_custom()` 函数形式提供：

```python
def add_custom(x, y):
    """
    实现自定义加法操作。
    
    参数:
        x (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        y (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        
    返回:
        Tensor: 计算结果张量，数据类型与输入一致，数据格式支持ND。
    
    注意:
        张量数据格式支持ND
    """

## 使用案例

import torch
import kernel_gen_ops

# 创建输入张量
x = torch.randn(8, 2048, dtype=torch.float16)
y = torch.randn(8, 2048, dtype=torch.float16)

# 使用add_custom执行计算
result = kernel_gen_ops.add_custom(x, y)

## 约束与限制

- 输入输出张量数据格式支持ND。
```

#### b. 代码框架与待填补信息 (来自 `tasks/.../question/` 目录)

为了引导 LLM 生成结构化的代码，框架会提供代码模板（骨架）。这些模板位于具体任务的 `question/` 目录下，通常包含：

* （当前支持）
    * `op_kernel/<operator_name>.cpp`: Kernel（Device端）代码模板。
* （未来支持）
    * `op_host/<operator_name>.cpp`: Host端代码模板。
    * `op_host/<operator_name>_tiling.h`: (如果需要) Tiling策略的头文件模板。

目前，LLM 仅需完善 op_kernel/<operator_name>.cpp 文件中的待填充内容，后两类模板将在未来逐步支持。

#### c. 通用指令与角色设定

在 Prompt 的最开始，通常会包含一些通用指令，例如：

* **角色扮演：** "你是一个专业的昇腾的算子开发工程师..."
* **目标语言与API：** "请使用 Ascend C 进行开发..."
* **代码风格要求：** "请确保代码可读性高，添加必要的注释..."
* **输出格式要求：** "请为 Kernel 和 Host 端分别生成代码。确保 Kernel 部分包含 `<kernel_name>_kernel` 函数，Host 部分包含 `Operator<OperatorName>Paras` 结构体和 `REGISTER_OP` 等宏。"
* **关键信息提示：** "请注意处理张量的数据类型、形状和布局。"


#### d. Prompt 组合逻辑

实际的 Prompt 生成逻辑主要位于 `kernel_generator/generate_codes_with_sft.py`  和 `kernel_generator/llm_api.py` (负责与 LLM API 交互) 中。大致流程是：
1.  读取目标任务的 `api_desc.md`。
2.  读取任务的 `question/` 目录下的代码模板。
3.  将上述信息与通用指令结合，构造成一个完整的 Prompt 文本发送给 (可能经过SFT的) LLM。

**核心思想：** 通过 `api_desc.md` 让模型理解“做什么”，通过代码模板让模型理解“输出的结构框架”，并通过通用指令设定行为和目标。

---

### 2. 模型需要生成/写哪些内容

在理解了上述 Prompt 后，大语言模型被期望生成以下内容，这些内容通常会替换或填入 `question/` 目录下的模板文件，最终形成一套完整的算子代码（通常保存在 `answer/` 或指定输出目录）：
当前主要支持 Kernel 端代码的生成，Host 端代码与 Tiling 头文件将在未来逐步支持。

#### a. Kernel 端代码 (`op_kernel/<operator_name>.cpp`)

这是在 NPU 设备上实际执行的算子逻辑。模型需要：
* 定义 Kernel 函数（例如，遵循 `extern "C" __global__ __aicore__ void <kernel_name>(...)` 的 Ascend C 规范）。
* 实现核心的数学运算和算法逻辑，如循环、数据搬运、计算指令等。
* 正确使用 Ascend C API（如 '....' 等）
* 处理张量数据，包括数据加载、存储、类型转换（如果需要）。
* 根据 `api_desc.md` 中的描述处理不同的数据类型和形状。

#### b. Host 端代码 (`op_host/<operator_name>.cpp`)

这是在 CPU（Host）上运行的代码，负责准备数据、调用 Kernel 并处理结果。模型需要：
* 实现算子注册相关的宏和函数（如 `REGISTER_OP(<OperatorName>).INPUT(...).OUTPUT(...).ATTR(...).OP_END_FACTORY_REG(...)`）。
* 定义参数结构体 (`Operator<OperatorName>Paras`) 用于解析和传递算子属性。
* 实现 `InferShape` 函数，根据输入推导输出形状。
* 实现 Tiling 函数的调用或 Tiling 逻辑本身，用于计算 Kernel 的分块信息和参数。
* (对于 Ascend C) 实现 Kernel Launch 相关的逻辑。

#### c. Tiling 头文件 (`op_host/<operator_name>_tiling.h`，如果适用)

对于一些复杂的 Ascend C 算子，Tiling 逻辑可能比较复杂，会单独放在一个头文件中。模型可能需要：
* 定义 Tiling 结构体或类。
* 实现 `DoTiling` 方法，根据输入形状、数据类型和硬件资源计算出合理的 Tiling 参数（如 `blockDim`, ` BufferNum` 等）。


**总结来说，模型的核心任务是根据 `api_desc.md` 的语义描述和代码模板的结构引导，生成一套功能正确、语法合规、可以被框架编译和运行的昇腾算子 C++ 代码（包括 Kernel 和 Host 两部分，以及可能的 Tiling 逻辑）。其完成此任务的能力，可能部分源于其基础训练，部分源于针对此类任务的特定SFT。**
