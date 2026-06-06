# NPUKernelBench V2.0: Benchmarking LLM-Driven AscendC Kernel Generation

[[License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

NPUKernelBench is a benchmark and evaluation framework for studying automated operator generation on Huawei Ascend NPUs. The benchmark focuses on AscendC kernel synthesis, compilation, numerical validation, and performance-oriented analysis under realistic operator-development constraints.

The V2.0 release extends the feasibility-oriented V1.0 benchmark with a larger operator suite and a domain-knowledge-injected model trained on high-quality Chain-of-Thought (CoT) supervision. The model is designed to internalize AscendC programming conventions, hardware-aware tiling strategies, and operator implementation patterns. Given natural-language requirements or formal operator descriptions, it can generate more than 50 usable AscendC kernels, representing a 400% increase over V1.0 and a substantial improvement in implementation quality and practical usability.

## Core Contributions

- **Hierarchical benchmark suite.** Operator tasks are organized by difficulty level, operator category, and operator name, enabling systematic evaluation across elementary, composite, and complex kernels.
- **LLM-based kernel generation.** The framework provides prompt construction, model interaction, code parsing, and generation utilities for AscendC kernel synthesis.
- **Automated evaluation pipeline.** End-to-end scripts manage code generation, compilation, numerical correctness checks, and performance measurement.
- **Reference-backed validation.** Each task includes a PyTorch reference implementation and task-specific test cases to support reproducible correctness evaluation.
- **Research-oriented documentation.** The documentation describes task design, scoring rules, benchmark execution, and model-serving workflows.

## Benchmark Design

All benchmark tasks are stored under `tasks/` and follow the structure:

```text
level/Category/OperatorName/
```

Each operator task contains:

- `question/`: implementation templates and task inputs for code generation.
- `answer/`: reference implementations used as official examples or baselines.
- `validation/`: PyTorch references, input generation logic, and test cases.
- `api_desc.md`: operator semantics, interface definitions, formulas, and implementation constraints.

### Difficulty Levels

- `level1`: elementary operators with simple data flow, usually single-input or element-wise operations such as `Sqrt`, `Equal`, and comparison operators.
- `level2`: common composite operators with multiple inputs, fused computation patterns, or nontrivial data movement such as `AddLayerNorm` and `GeluGrad`.
- `level3`: advanced operators involving dynamic shapes, complex parallelization, global data dependencies, or specialized orchestration such as `TopKV3` and `BasicMatmul`.

For details, see [Benchmark Task Design](./docs/BENCHMARK_TASKS.md).

## Evaluation Methodology

The evaluation pipeline reflects the standard lifecycle of AscendC operator development:

1. **Compilation correctness.** The generated implementation must compile successfully with the CANN toolchain.
2. **Numerical correctness.** The NPU output is compared with the PyTorch reference output under task-specific tolerance thresholds. The default preliminary tolerance is `1e-3` for both absolute and relative errors unless a task defines a custom checker.
3. **Performance measurement.** For implementations that satisfy correctness constraints, runtime latency and normalized throughput metrics can be collected on Ascend hardware.

For details, see [Benchmark Evaluation](./docs/BENCHMARK_EVALUATION.md).

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/weich97/NPUKernelBench.git
cd NPUKernelBench
```

### 2. Configure the framework environment

```bash
source set_framework_env.sh
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Install vLLM-Ascend

Follow the official vLLM-Ascend installation guide:

https://docs.vllm.ai/projects/ascend/en/latest/installation.html

The framework expects vLLM `0.7.3-dev` or later.

### 5. Start the vLLM service

Set `chat.model_path` in `base_config.yaml` to the model checkpoint path, then launch the serving process:

```bash
nohup bash start_vllm_server.sh > vllm_server.log 2>&1 &
```

For additional configuration details, see [Start vLLM Server](./docs/START_VLLM_SERVER.md).

## Example Run

The following command evaluates the official reference answers for the `Sqrt` and `SwiGlu` tasks:

```bash
python run_multi_test.py -chat -task_name Sqrt SwiGlu
```

The terminal output reports generation, compilation, and numerical validation progress for each selected operator.

## Typical Workflow

![NPUKernelBench workflow](figures/workflow.png)

1. **Select a task.** Choose one or more operator tasks, for example `tasks/level1/Math/Sqrt`.
2. **Generate or provide an implementation.** Use the LLM generation path or manually implement the AscendC kernel.
3. **Run evaluation.** Execute the compilation and correctness stages with `run_multi_test.py`.
4. **Inspect outputs.** Generated source code, build artifacts, and logs are stored under the configured run directory.

### LLM-Based Generation

```bash
python run_multi_test.py -chat -task_name Sqrt -stages code_gen
```

Generated samples are stored under a path similar to:

```text
runs/msopgen/lvl1/Math/Sqrt/fixed_case_0/sample0/
```

By default, the framework can generate multiple candidate kernels for a static-shape test case using a kernel-only template. See [LLM Kernel Generation](./docs/LLM_KERNEL_GENERATION.md) for details.

### Evaluation of Generated Code

```bash
python run_multi_test.py -chat -task_name Sqrt -stages compile precision
```

### Evaluation of Manually Written Code

```bash
python run_multi_test.py -task_name Sqrt -stages compile precision
```

## Repository Structure

```text
NPUKernelBench/
|-- docs/                    # Benchmark documentation and user guides
|-- figures/                 # Workflow and case-study figures
|-- framework/               # Core automation and evaluation framework
|-- kernel_generator/        # Prompting and LLM-based code generation utilities
|-- libs/                    # Shared Ascend/CANN support libraries
|-- tasks/                   # Benchmark task suite
|-- base_config.yaml         # Global configuration
|-- run_multi_test.py        # Main execution entry point
|-- set_framework_env.sh     # Environment setup script
`-- start_vllm_server.sh     # vLLM service launch script
```

## Case Studies

### Knowledge Injection for AscendC Instruction Use

![Knowledge injection effect](figures/knowledge_injection_effect.png)

The first case study examines use of the AscendC `Muls` instruction. Before domain adaptation, the model response is uncertain, relies on generic programming analogies, and fails to identify the correct AscendC API usage. After CoT-based domain-knowledge injection, the model provides structured reasoning about vector multiplication, source and destination operands, data-layout constraints, queue management, data movement, and scalar-value invocation.

### Tiling-Aware Swish Kernel Generation

![Comparison of Swish implementations](figures/compare_swish_impl.png)

The second case study analyzes an AscendC implementation of the `Swish` operator. The adapted model correctly identifies the mathematical form `y = x * sigmoid(x)`, designs a remainder-aware multicore tiling strategy, and composes AscendC primitives such as `Muls`, `Exp`, `Adds`, and `Div` into a valid implementation pipeline.

## License

This project is released under the [Apache 2.0 License](LICENSE).
