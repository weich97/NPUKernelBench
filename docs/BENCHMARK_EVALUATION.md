# NPUKernelBench 评测体系

## 1. 简介

NPUKernelBench 是一个专为全面评估 NPU（神经网络处理器）自定义内核算子（Kernel）能力而设计的基准测试套件。本项目旨在通过结构化、多层次的评测系统，深入衡量算子在**正确性**和**性能**两大核心维度的表现。

本评测体系强调，在追求卓越性能的同时，算子的**正确性必须达到 100%**。其核心目标是构建一个公平、可复现且具有挑战性的评测基准，以期引导开发者编写出兼具高效率和高正确性的算子代码，从而充分释放 NPU 硬件的潜在性能。

## 2. 设计理念

本项目构建此 Benchmark 的核心设计理念如下：

### 正确性优先 (Correctness First)
算子功能的正确性是其存在的基石。任何精度问题，无论其大小，都将导致在评分上的"一票否决"。一个无法保证在所有场景下都输出正确结果的算子，其性能表现毫无意义。

### 性能为最终目标 (Performance as the Ultimate Goal)
在确保 100% 正确的前提下，我们鼓励开发者挑战硬件的性能极限。**当前版本的 Benchmark 专注于正确性评估，而未来的第二阶段将引入全面的性能评分体系**，旨在对标业界顶尖实现，引导开发者写出兼顾准确与高效的算子。

### 分级挑战 (Multi-level Difficulty)
本Benchmark 包含从 `Level 1` 到 `Level 3` 的多级别任务，旨在全面评估开发者从入门到精通的算子开发能力：

- **Level 1**: **基础算子**。主要包含单步、基础的算子实现，例如 `Add`、`Sqrt` 等。此级别旨在考察开发者对算子开发基础流程和关键接口的掌握情况
- **Level 2**: **通用算子**。包含在各类模型中广泛使用且优化需求明确的通用算子，如 `LayerNorm`、`Gelu` 等。此级别旨在考察开发者对 Tiling、数据搬运等关键优化技巧的运用能力
- **Level 3**: **复杂算子**。主要为需要复杂融合优化或与特定模型架构强相关的算子，例如 `TopK`、`GroupedMatmul`、`PagedAttention` 等。此级别对开发者的综合优化能力、算法理解深度和性能调试经验提出了更高要求

在最终计分时，我们将赋予高难度等级的任务更高的权重。

## 3. 评测维度

基于上述原则，我们将评测体系具体化为以下两个维度：

### 精度评测 (Correctness Evaluation)
- **机制**：针对于每个算子任务，我们预设了多个测试用例（Test Case），全面覆盖不同的数据类型和张量形状（Shape）组合。评测框架将自动编译并执行用户提交的算子，随后将其输出结果与预先生成的"黄金参考"（Golden Reference）输出进行逐位比对
- **标准**：仅当一个测试用例的输出**满足预设的精度标准**时，方可被认定为通过。整个算子任务的精度将以**通过率**的形式体现

### 性能评测 (Performance Evaluation)
- **注意：性能评测为 Benchmark 的第二阶段目标，当前版本的打分体系尚未纳入性能得分**
- **指标**：性能评测的核心指标为算子在 NPU 上执行的**运行延迟 (Execution Latency)**。为确保评测的公平性与稳定性，我们将进行多次预热（Warmup）后，连续运行 N 次，取**平均执行时间**作为该测试用例下的最终性能耗时
- **基准**：性能的优劣通过与一个客观的"性能上限"基准进行比较来衡量。此基准我们定义为 **T_roofline**
- **T_roofline 的定义**：代表了在当前硬件上，该算子及特定 Shape 组合所能达到的理论最快运行时间。在实践中，它将优先采用 **NPU 厂商官方提供的高性能库（如 `aclnn`）中的同类算子实现**作为其有力的代理基准

## 4. 评分系统

我们的评分系统将根据评测体系的发展，分两个阶段实施：

- **第一阶段（当前已实现）：正确性评分**。此阶段全面聚焦算子代码的功能正确性、运行稳定性与鲁棒性。所有评分均基于精度测试的结果
- **第二阶段（未来规划）：性能评分**。在成功通过第一阶段的正确性考验后，将引入性能指标，对算子的运行效率进行量化评估

**以下将简述第一阶段的评分体系。**

### 静态与动态 Shape 任务说明

我们的评分体系在设计上能统一处理静态与动态 Shape 任务，但在评估的"单元"上有所区别：

- **静态 Shape 任务**: 在此类任务中，一个算子任务（Task）仅针对**一个特定的 Shape**。因此，该任务下所有的测试用例均服务于这一个 Shape 的精度评估
- **动态 Shape 任务**: 在此类任务中，一个算子任务（Task）需要提交一个**能够处理多种 Shape** 的通用 Kernel。因此，该任务下的测试用例会包含 M 个不同的 Shape，用以综合评估这一个通用 Kernel 的**鲁棒性（精度）**

这种设计使得任务得分（Score_task）自然地成为了动态算子**通用性**的量化指标。一个无法处理所有规定 Shape 的动态算子，其分数会相应降低。

### 4.1 单个任务得分 (Score_task)

一个算子任务（Task）的总分，完全由其在该任务下所有测试用例的**整体精度表现**决定，用于衡量算子的稳定性和鲁棒性。

**公式**:
$$Score_{task} = \frac{\text{Number of Passed Test Cases}}{\text{Total Number of Test Cases}} \times 100$$

**设计解读**:
- 此分数直接反映了算子的正确性
- 如果任务精度达到 100%（即所有测试用例通过），则任务得分为 100
- 如果任务精度为 80%（例如动态算子未能通过部分 Shape），其得分则为 80
- 如果任务精度为 0，则任务总分直接为 0，体现**正确性优先**的原则

### 4.2 等级得分 (Score_level)

每个等级（L1, L2, L3）的得分是该等级下所有任务得分的**算术平均值**。

**公式**:
$$Score_{L1} = \text{Average}(Score_{task\_i})_{\text{for all tasks } i \text{ in L1}}$$

（Score_L2, Score_L3 同理）

### 4.3 最终总分 (Score_final)

最终总分是三个等级得分的**加权和**，权重旨在体现不同等级的难度与重要性。

**公式**:
$$Score_{final} = (Score_{L1} \times w_{L1}) + (Score_{L2} \times w_{L2}) + (Score_{L3} \times w_{L3})$$

**建议权重 (可配置)**:
- w_L1 = 0.2
- w_L2 = 0.3
- w_L3 = 0.5

**说明**：权重向高难度等级倾斜，旨在鼓励开发者挑战并解决更为复杂的问题。

### 评分公式汇总 (Stage 1: Correctness)

| 评分项 | 计算公式 | 核心思想               |
|:------|:--------|:-------------------|
| **任务得分** | Score_task = (通过用例数/总用例数) × 100 | 衡量精度正确性，是得分的基石     |
| **等级得分** | Average(Score_task_j) | 体现该等级下的综合平均能力      |
| **最终总分** | Σ(Score_level × w_level) | **加权求和，激励挑战高难度任务** |

## 5. Benchmark 执行、配置与评估

本章将详细说明如何运行 Benchmark，理解评测流程，配置相关参数，并解读输出结果与目录结构。

### 5.1 快速上手：运行 Benchmark

#### 5.1.1 执行完整的 Benchmark 套件

要启动对所有已定义算子任务的全面评测，请在项目根目录下执行以下命令：

```bash
python run_multi_test.py -chat [-stages code_gen compile precision perf]
```

方括号中的内容为可选参数，默认的 stages 包含 code_gen、compile、precision 三个阶段。

| stage | 功能说明 |
|:------|:--------|
| **code_gen** | 代码生成 |
| **compile** | 编译 |
| **precision** | 精度测试 |
| **perf** | 性能测试 |

执行该命令后，系统将开始处理 `tasks/` 目录中定义的所有算子任务。评测完成后，终端将打印统计信息，其格式如下所示：

```
Detailed Statistics by Operator Level:

[Statistics for Level 1]

+--------------------+------------+-----------+-----------------+---------------------+----------------+------------------+
| op_definition      |   op_level |   case_id |   total_samples |   format_parse_pass |   compile_pass |   precision_pass |
+====================+============+===========+=================+=====================+================+==================+
| Math/Sqrt          |          1 |         0 |               5 |                   5 |              4 |                2 |
+--------------------+------------+-----------+-----------------+---------------------+----------------+------------------+
| TensorCreation/Eye |          1 |         0 |               5 |                   5 |              3 |                0 |
+--------------------+------------+-----------+-----------------+---------------------+----------------+------------------+


[Statistics for Level 2]

+-------------------+------------+-----------+-----------------+---------------------+----------------+------------------+
| op_definition     |   op_level |   case_id |   total_samples |   format_parse_pass |   compile_pass |   precision_pass |
+===================+============+===========+=================+=====================+================+==================+
| Loss/MseLoss      |          2 |         0 |               5 |                   5 |              0 |                0 |
+-------------------+------------+-----------+-----------------+---------------------+----------------+------------------+
| Norm/AddLayerNorm |          2 |         0 |               5 |                   5 |              0 |                0 |
+-------------------+------------+-----------+-----------------+---------------------+----------------+------------------+

[Statistics for Level 3]

+-----------------+------------+-----------+-----------------+---------------------+----------------+------------------+
| op_definition   |   op_level |   case_id |   total_samples |   format_parse_pass |   compile_pass |   precision_pass |
+=================+============+===========+=================+=====================+================+==================+
| GMM/BasicMatmul |          3 |         0 |               5 |                   5 |              0 |                0 |
+-----------------+------------+-----------+-----------------+---------------------+----------------+------------------+

Overall Statistics Summary:

+------------+-----------------+-------------------+---------------------+-----------------------+
|   op_level |   total_samples | parse_pass_rate   | compile_pass_rate   | precision_pass_rate   |
+============+=================+===================+=====================+=======================+
|          1 |              10 | 100.0%            | 70.0%               | 20.0%                 |
+------------+-----------------+-------------------+---------------------+-----------------------+
|          2 |              10 | 100.0%            | 0.0%                | 0.0%                  |
+------------+-----------------+-------------------+---------------------+-----------------------+
|          3 |               5 | 100.0%            | 0.0%                | 0.0%                  |
+------------+-----------------+-------------------+---------------------+-----------------------+

[The NPUKernelBench benchmark achieved a total score of: 0.04]
```

**结果解读：**

执行 Benchmark 后，您将看到两部分统计信息：`Statistics for Level X` 和 `Overall Statistics Summary`:

- **`[Statistics for Level X]` 部分**：
  - 该部分展示了按算子等级（`op_level`，对应第 2 章的分类）分组的详细结果，具体如下：
  - `op_definition`: 算子定义的名称（例如 `Math/Sqrt`）
  - `op_level`: 该算子所属的复杂度等级
  - `case_id`: 根据特定的测试用例生成算子（仅在静态 shape 测试模式下适用，可在 `base_config.yaml` 中配置 `n_case` 数值，以生成多个特定测试用例。其中 `case_id` 与测试文件 `tasks/level1/Math/Sqrt/validation/test_cases.csv` 的内容相对应）
  - `total_samples`: 针对此算子及其特定测试用例，模型生成 Kernel 代码的尝试次数，即 rollout 次数。此数值可在 `base_config.yaml` 中配置 `n_sample`
  - `format_parse_pass`: 在 `total_samples` 次尝试中，模型生成的代码符合预设格式并成功解析的次数
  - `compile_pass`: 在 `total_samples` 次尝试中，成功通过昇腾 C 编译器编译的次数
  - `precision_pass`: 在 `total_samples` 次尝试中，成功通过精度验证的次数

- **`Overall Statistics Summary:` 部分**：
  - 该部分提供了每个算子等级的汇总统计数据
  - `op_level`: 算子复杂度等级
  - `total_samples`: 该等级下包含的算子测试实例总数。例如，如果等级 1 有 2 个不同的 `op_definition`，每个 `op_definition` 有 1 个特定测试用例 `n_case=1`，并进行 5 次生成尝试 `n_sample=5`，则此处的 `total_samples` 为 10
  - `parse_pass_rate`: 该等级所有测试实例的 `format_parse_pass` 总和除以 `total_samples` (rollouts) 总和的百分比
  - `compile_pass_rate`: 该等级所有测试实例的 `compile_pass` 总和除以 `total_samples` (rollouts) 总和的百分比
  - `precision_pass_rate`: 该等级所有测试实例的 `precision_pass` 总和除以 `total_samples` (rollouts) 总和的百分比

#### 5.1.2 分析特定算子

如果希望仅针对某一个或某几个特定的算子进行分析和测试，可以使用 `-task_name` 参数。框架将自动在所有已定义的任务中查找并执行指定的算子。

例如，仅测试名为 `Sqrt` 的算子，执行以下命令：

```bash
python run_multi_test.py -chat -task_name Sqrt
```

如果需要测试多个特定算子，可用空格将它们分隔开：

```bash
python run_multi_test.py -chat -task_name Sqrt Eye MseLoss AddLayerNorm BasicMatmul
```

### 5.2 评测工作流与核心指标

Benchmark 的评测流程高度自动化，旨在提供客观、一致的算子质量评估。

#### 5.2.1 自动化评测管线

整个评测过程主要包括以下阶段，这些阶段会尽可能并行处理以提高效率：

1. **算子生成**: 框架会根据任务定义，并行调用大语言模型（例如通过 VLLM API）为每个任务的每个特定测试用例生成指定数量（`n_sample`）的算子 Kernel 代码。生成的代码会存放在配置的 `src_dir`（通常在 `run_dir` 下的特定任务目录中）

2. **编译评估**: 对所有生成的算子代码样本进行并行编译

3. **精度评测**: 根据可用的硬件资源（如 NPU 卡数），并行执行精度测试

在每个阶段，框架都会在控制台打印各个算子测试用例的进展情况以及相关日志文件的路径。

#### 5.2.2 编译评估

此阶段检查 AI 生成的算子代码是否符合昇腾 C 的语法规范，并能够成功构建。

- **过程**: 对每个生成的代码样本调用昇腾 C 编译器
- **输出**: 编译成功或失败的状态。详细的编译日志会保存到每个算子样本的 `log/` 目录下（可以查阅对应的 `log_file` 以定位编译错误）

例如：`runs/msopgen/lvl1/Math/Sqrt/fixed_case_0/sample0/log/lvl1_categoryMath_Sqrt_sample0_compile.log`

#### 5.2.3 精度评估

精度评估是衡量 AI 生成算子功能正确性的核心环节。

- 对于每个算子任务，其 `validation/module.py` 文件中通常定义了两个 PyTorch Module：
  - `Model`: 通过调用 PyTorch 官方 API 或其他可靠方式实现算子功能的参考模型
  - `ModelNew`: 通过 Pybind C++ 扩展调用由 AI 生成的、已编译的昇腾 C 算子

- **默认常规评测标准**
  - 使用相同的输入数据分别执行 `Model` 和 `ModelNew`，然后比较它们的输出，比较标准同时考虑绝对误差和相对误差（可在 `base_config.yaml` 中配置 `max_abs_error` 和 `max_rel_error`）
  - 对于有多个输出张量的算子，仅当所有输出张量均满足精度要求，该算子才被认为精度达标

```python
def check_precision(outputs, outputs_new, max_abs_error, max_rel_error):
    # outputs: list of tensors from reference model
    # outputs_new: list of tensors from AI-generated kernel
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new
    
    all_abs_diff, all_rel_diff = [], []
    is_accurate = True
    
    # Process each output pair
    for out, out_new in zip(outputs, outputs_new):
        if out_new.dtype == torch.bool:
            out = out.to(torch.int)
            out_new = out_new.to(torch.int)
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        all_abs_diff.append(abs_diff.reshape(-1))
        all_rel_diff.append(rel_diff.reshape(-1))

        if torch.any(~torch.isfinite(out)) or torch.any(~torch.isfinite(out_new)) or torch.any(~torch.isfinite(abs_diff)) or torch.any(~torch.isfinite(rel_diff)):
            is_accurate = False

        # Check if within precision requirements
        if ((abs_diff > max_abs_error) & (rel_diff > max_rel_error)).any():
            is_accurate = False
    
    # Combine all differences
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    
    return (1 if is_accurate else 0), all_abs_diff, all_rel_diff
```

- **自定义算子精度评估标准**
  - 可根据不同数据类型或者算子自身精度要求自定义精度标准，在相应算子的 `validation/prepare_inputs.py` 文件中添加 `custom_check_precision` 函数
  - 示例参考（`tasks/level2/Math/SwiGlu/validation/prepare_inputs.py`）：

```python
def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float32:
        return check_precision(outputs, outputs_new, max_abs_error=0.00001, max_rel_error=0.00001)
    else:
        return check_precision(outputs, outputs_new, max_abs_error=0.001, max_rel_error=0.001)
```

- **输出**: 精度测试的结果，包括是否成功、成功与失败的案例数、详细日志文件路径，以及每个测试用例（case）的平均/最大绝对误差、平均/最大相对误差等

#### 5.2.4 性能评估

除了功能正确性，算子性能也是关键考量。性能评测模块负责在真实的昇腾硬件环境下执行 AI 生成的算子，并采集其运行时间等指标，以便与参考实现进行对比。相关参数如性能测试的试验次数 (`num_perf_trials`) 可以在 `base_config.yaml` 中配置。详细的性能数据通常记录在日志中，可用于与参考实现进行比较分析。

### 5.3 配置 Benchmark (`base_config.yaml`)

`base_config.yaml` 文件是 Benchmark 的核心配置文件，允许用户调整评测流程的多个方面。以下是一些关键参数的说明：

- **目录设置**:
  - `run_dir: "runs"`: 指定所有任务的模型执行结果（包括生成的 Kernel 代码、编译输出、日志等）的根存储路径
  - `src_dir: "src"`: 在每个具体任务的 `run_dir` 下，存放模型生成的算子源码的子目录名
  - `log_dir: "log"`: 存放详细日志文件的子目录名
  - `build_dir: "build"`: 存放算子最终编译结果的子目录名

- **执行控制**:
  - `encoding: "utf-8"`: 文件编码
  - `n_sample: 100`: 对于一个任务的一个测试用例（case），模型需要执行多少次代码生成尝试（rollout）
  - `n_case: 1`: 每个算子任务默认测试多少个用例（case）。一个 case 对应 `test_cases.csv` 文件中的一行，通常是一个特定的 shape（仅在静态 shape 测试场景内适应）
  - `show_result_log_file: False`: 是否在控制台输出中显式打印结果日志文件的完整内容
  - `static_shape_mode: True`: 是否启用静态 shape 测试模式（分为静态、动态 shape 测试两种模式）
  - `kernel_only_mode: True`: 是否启用 kernel only 模版（分为kernel only / full 两种模版）

- **LLM API 设置 (`chat`)**:
  - `api_model: "vllm"`: 指定使用的 API 模型类型
  - `api_max_retries: 3`: API 调用失败时的最大重试次数
  - `api_retry_delay: 5`: API 调用重试的延迟时间（秒）
  - `model_path: "/XXX/qwen3-32b/qwen3_ep5"`: VLLM 等本地模型的路径
  - `generate_new_answer: True`: 是否强制重新生成答案，即使已有缓存
  - `parse_new_answer: True`: 是否强制重新解析答案

- **编译设置 (`compile`)**:
  - `mode: "msopgen"`: 采用的编译模式
  - `common_dir: "libs/common_include"`: 编译时可能需要的公共头文件目录
  - `num_processes: 128`: 编译时并发的最大进程数目

- **评估设置 (`eval`)**:
  - `verbose: True`: 是否输出详细的评估信息
  - `gpu_devices:`: 指定用于精度和性能测试的设备 ID 列表
  - **性能相关**:
    - `num_perf_trials: 100`: 性能测试时，每个算子执行的轮次，用于取平均值或中位数以获得稳定性能数据
  - **精度相关**:
    - `num_correct_trials: 1`: 每个算子的每个 case 进行精度测试的次数（通常为 1 即可，因为精度是确定性的）
    - `max_rel_error: 0.001`: 可接受的最大相对误差
    - `max_abs_error: 0.001`: 可接受的最大绝对误差

### 5.4 理解输出目录结构

所有评测运行的结果都会被组织存放在 `run_dir`（默认为 `runs/`）指定的目录下。其内部结构通常如下：

```
<run_dir>/
    `-- msopgen/                                # 固定的编译选项目录名
        |-- lvl1                                # 算子等级目录
        |   `-- Math                            # 类别组织目录
        |       `-- Sqrt                        # 具体的算子名称
        |           |-- fix_case_0              # 特定测试用例（仅在静态 shape 测试模式下适用）
        |           |   |-- sample0             # 针对该算子的一个测试样本/配置
        |           |   |   |-- build/          # 存放编译产物 (如 .so 文件)
        |           |   |   |-- log/            # 存放详细日志 (生成、编译、精度、性能)
        |           |   |   `-- src/            # 存放AI模型生成的Kernel源码 (如 .cpp 文件)
        |           |   `-- sample1
        |           |       |-- ...
        |           `-- fix_case_1
        |               |-- ...
        `-- lvl2
            |-- Activation
            |   |-- GeGluV2
            |   |   `-- fixed_case_0 
            |   |       `-- sample0
            |   |           |-- build/
            |   |           |-- log/
            |   |           `-- src/
            |   `-- SwiGlu
            |       `-- fixed_case_0
            |           `-- sample0
            |               |-- build/
            |               |-- log/
            |               `-- src/
            `-- Norm
                `-- AddLayerNorm
                    `-- fixed_case_0        
                        `-- sample0
                            |-- build/
                            |-- log/
                            `-- src/
```

- **`<run_dir>/msopgen/<level>/<category>/<OperatorName>/fix_case_<N>/sample<N>/`**: 这是针对特定算子、特定测试用例、特定样本（如果一个算子有多次不同的实现尝试，会用 `sample0`, `sample1`, ... 区分，这对应于 `n_sample` 的不同 rollout）的独立工作目录
  - `src/`: 包含 AI 模型为该样本生成的所有相关源文件，主要是 Kernel 代码
  - `build/`: 包含编译此样本代码后产生的中间文件和最终库文件
  - `log/`: 包含与此样本相关的最详细日志，包括代码生成阶段的 prompt 和回复、完整的编译输出、精度测试的详细步骤和中间值、以及性能测试的原始数据，可用于分析模型行为和调试失败案例