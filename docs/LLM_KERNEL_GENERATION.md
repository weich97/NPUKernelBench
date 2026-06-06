# LLM Kernel Generation

The kernel-generation module uses large language models to synthesize AscendC operator implementations from task descriptions, test-case metadata, hardware information, and code templates. Its objective is to produce structured source files that can be compiled and evaluated by the NPUKernelBench framework.

## 1. Prompt Structure

A generation prompt is assembled from the following components.

### 1.1 Operator Specification (`api_desc.md`)

Each task contains an `api_desc.md` file that defines the operator semantics and interface. It typically includes:

- mathematical or logical functionality;
- input and output tensor names, data types, shapes, and layouts;
- operator attributes and constraints;
- implementation notes relevant to AscendC;
- formulas or reference equations when applicable.

Example specification fragment:

```markdown
# aclnnBasicMatmul

## Functional Description

This AscendC operator performs matrix multiplication between two two-dimensional tensors, `A` and `B`. It is a fundamental primitive for linear layers and attention-style computations.

## Formula

Given `A` with shape `m x k` and `B` with shape `k x n`, the output `C` has shape `m x n`:

$$C_{ij} = \sum_{p=1}^{k} A_{ip} B_{pj}$$
```

### 1.2 Code Templates (`question/`)

Task templates are located under the `question/` directory. They usually contain:

- `op_kernel/<operator_name>.cpp`: device-side AscendC kernel template;
- `op_host/<operator_name>.cpp`: host-side registration, shape inference, launch, and tiling logic;
- `op_host/<operator_name>_tiling.h`: optional tiling data structures.

The framework supports two modes:

- `kernel_only_mode=True`: the model fills only the kernel-side template;
- `kernel_only_mode=False`: the model fills both host-side and kernel-side templates.

### 1.3 General Instructions

The prompt also includes role and format instructions. These instructions define the model as an AscendC expert, require syntactically valid and executable code, preserve existing function names and class definitions, and constrain the output to XML-like file blocks that can be parsed by the framework.

### 1.4 Prompt Assembly

The prompt assembly logic is implemented primarily in `kernel_generator/generate_codes_with_sft.py` and `kernel_generator/llm_api.py`.

The process is:

1. read `api_desc.md` for the target task;
2. read `test_cases.csv` and relevant input-preparation code;
3. read the operator templates from `question/`;
4. append hardware information and generation constraints;
5. send the final prompt to the configured model backend.

## 2. Expected Model Output

The model is expected to generate complete source content for the files specified in the output template.

### 2.1 Kernel-Side Code

The kernel implementation must:

- define the required AscendC kernel entry point;
- implement the mathematical or logical operator semantics;
- use AscendC APIs correctly for data movement, vector instructions, and type conversion;
- respect alignment and memory constraints;
- handle the data types and shapes described by the task specification and test cases.

### 2.2 Host-Side Code

When full-template mode is enabled, the host-side implementation must:

- define and populate operator parameter structures;
- implement shape inference where required;
- compute tiling parameters and launch configuration;
- call the kernel with correct block dimensions and runtime arguments.

### 2.3 Tiling Headers

For operators with nontrivial tiling, the generated code may also need to define tiling structures, serialization logic, and helper fields in a header file.

## 3. Generation Quality Criteria

A generated implementation is considered useful only if it satisfies all of the following requirements:

- it can be parsed into the expected file structure;
- it compiles under the selected CANN/MSOpGen backend;
- it passes the numerical validation cases;
- it follows AscendC data-alignment and memory-access requirements;
- it avoids undefined symbols, unsupported type names, or changes to protected template interfaces.

These criteria allow NPUKernelBench to study not only whether a model can write code, but whether it can produce deployable NPU kernel implementations under realistic constraints.
