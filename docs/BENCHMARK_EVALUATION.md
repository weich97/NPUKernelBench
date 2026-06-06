# Benchmark Evaluation Methodology

NPUKernelBench evaluates generated AscendC kernels along the core stages of operator development: compilation, numerical correctness, and performance measurement. Correctness is the primary requirement; performance is meaningful only after an implementation produces valid outputs.

## 1. Evaluation Principles

### Correctness First

A kernel must produce correct outputs for all configured test cases before it can be considered a valid solution. Any numerical failure, runtime failure, or compilation failure invalidates the corresponding sample for the affected task.

### Performance Under Correctness Constraints

Performance measurements are used to analyze runtime efficiency after correctness has been established. The benchmark can record latency and related metrics for comparison against reference implementations or hardware-specific baselines.

### Multi-Level Difficulty

Tasks are grouped into levels from elementary to complex operators. Higher levels are intended to carry greater research significance because they require deeper algorithmic understanding, more sophisticated tiling, and stronger hardware-awareness.

## 2. Evaluation Stages

### 2.1 Code Generation

When `-chat` is enabled, the framework constructs prompts from `api_desc.md`, test-case metadata, hardware information, and code templates. Generated code is written into the configured run directory under per-task and per-sample folders.

### 2.2 Compilation

The compilation stage invokes the selected CANN/MSOpGen build path. A sample passes this stage only if all required source files compile successfully and the expected binary artifacts are produced.

### 2.3 Numerical Correctness

For each test case, the benchmark prepares inputs, executes the generated operator, and compares the result with a PyTorch reference implementation.

The default task score is:

$$Score_{task} = \frac{\text{Number of Passed Test Cases}}{\text{Total Number of Test Cases}} \times 100$$

A task receives full correctness credit only when all configured cases pass. Custom task-specific checkers may be defined in `validation/prepare_inputs.py` when an operator requires specialized tolerance handling.

Default tolerance parameters are configured in `base_config.yaml`:

- `eval.max_abs_error`: maximum allowed absolute error;
- `eval.max_rel_error`: maximum allowed relative error;
- `eval.num_correct_trials`: number of correctness trials per case.

### 2.4 Performance Measurement

The performance stage executes a valid operator on Ascend hardware and records runtime statistics such as average latency over repeated trials. The number of trials is controlled by `eval.num_perf_trials`.

Performance data are primarily used for comparative analysis and future scoring extensions. Correctness remains the prerequisite for interpreting performance results.

## 3. Aggregate Scores

For each level, the benchmark computes the arithmetic mean of task scores:

$$Score_{L1} = \text{Average}(Score_{task_i}) \quad \text{for all tasks } i \text{ in L1}$$

The same definition applies to `Score_L2` and `Score_L3`. A final weighted score can be computed as:

$$Score_{final} = Score_{L1} \times w_{L1} + Score_{L2} \times w_{L2} + Score_{L3} \times w_{L3}$$

A typical weighting scheme is:

- `w_L1 = 0.2`
- `w_L2 = 0.3`
- `w_L3 = 0.5`

These weights emphasize more complex operators while preserving visibility into elementary correctness.

## 4. Running the Benchmark

Evaluate generated code for a task:

```bash
python run_multi_test.py -chat -task_name Sqrt -stages compile precision
```

Evaluate manually written code:

```bash
python run_multi_test.py -task_name Sqrt -stages compile precision
```

Select multiple tasks:

```bash
python run_multi_test.py -chat -task_name Sqrt SwiGlu
```

## 5. Configuration

`base_config.yaml` controls the major benchmark parameters.

Directory fields:

- `run_dir`: root directory for generated code, builds, and logs;
- `src_dir`: generated source-code subdirectory;
- `build_dir`: build-artifact subdirectory;
- `log_dir`: detailed log subdirectory.

Execution fields:

- `n_sample`: number of generation attempts per task case;
- `n_case`: number of test cases to evaluate by default;
- `static_shape_mode`: whether static-shape mode is enabled;
- `kernel_only_mode`: whether the generator fills only kernel-side templates.

Model-serving fields under `chat`:

- `api_model`: API backend name;
- `model_path`: local model checkpoint path for vLLM-style serving;
- `temperature` and `max_tokens`: generation parameters;
- `timeout`: API timeout in seconds.

Compilation fields under `compile`:

- `mode`: compilation backend;
- `common_dir`: shared include/library directory;
- `num_processes`: maximum compilation parallelism.

Evaluation fields under `eval`:

- `gpu_devices`: devices used for correctness and performance evaluation;
- `num_perf_trials`: performance repetitions;
- `max_rel_error` and `max_abs_error`: default correctness tolerances;
- `timeout_perf` and `timeout_precision`: stage timeouts.

## 6. Output Layout

Benchmark outputs are organized under `run_dir`:

```text
runs/
`-- msopgen/
    `-- lvl1/
        `-- Math/
            `-- Sqrt/
                `-- fixed_case_0/
                    `-- sample0/
                        |-- build/   # compiled artifacts
                        |-- log/     # generation, compilation, precision, and performance logs
                        `-- src/     # generated source files
```

Each sample directory is an independent evaluation unit. The logs provide the primary evidence for diagnosing generation failures, compilation errors, numerical mismatches, and performance behavior.
