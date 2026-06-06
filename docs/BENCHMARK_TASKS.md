# Benchmark Task Design: Levels, Categories, and Shape Regimes

NPUKernelBench is designed to evaluate whether language models can synthesize correct and efficient AscendC kernels across a range of operator-development scenarios. A single aggregate success rate is insufficient: a model that can generate an element-wise addition kernel may still fail on attention-style data movement, and a model that can optimize a fixed-size input may not generalize to dynamic shapes.

The benchmark therefore decomposes operator difficulty into two orthogonal dimensions: algorithmic complexity and interface complexity.

## 1. Algorithmic Complexity

Algorithmic complexity describes the intrinsic computation performed by an operator. It determines the operator level.

### Level 1: Elementary Operators

Level 1 tasks contain simple mathematical or logical operations with mostly linear data flow.

Evaluation focus:

- understanding elementary arithmetic or logical semantics;
- generating syntactically valid and functionally complete AscendC code;
- handling basic data types, memory movement, and vectorized execution.

Typical characteristics:

- element-wise arithmetic, comparison, or logical operations;
- simple point-to-point mappings with limited data dependencies;
- little or no attribute-driven branching.

Representative examples include `Add`, `Sub`, `Mul`, `Equal`, `Cast`, `Fill`, and elementary activation operators.

### Level 2: Structured Operators

Level 2 tasks involve standard neural-network operators or composite kernels with nontrivial but bounded structure.

Evaluation focus:

- implementing algorithms with local data dependencies, such as windowed operations;
- adapting computation to operator attributes such as `stride`, `padding`, or `keep_dims`;
- maintaining numerical stability and reasonable data movement efficiency.

Typical characteristics:

- matrix multiplication, reductions, pooling, normalization, or common activation functions;
- windowed access patterns or aggregation along specific axes;
- finite and predictable control-flow variation driven by attributes.

Representative examples include `Conv2D`, `MaxPool2D`, `MatMul`, `ReduceSum`, `Sigmoid`, and `LayerNorm`.

### Level 3: Complex Operators

Level 3 tasks require complex algorithms, global data dependencies, dynamic control flow, or specialized numerical treatment.

Evaluation focus:

- implementing iterative, custom, or globally dependent algorithms;
- handling data-dependent control flow, sparse access, or synchronization;
- generating highly parallel code for complex memory and computation patterns;
- managing numerical sensitivity under hardware constraints.

Representative examples include `TopK`, `BasicMatmul` under complex regimes, `NMS`, `CTCGreedyDecoder`, attention-style kernels, and sparse segment operations.

## 2. Interface Complexity

Interface complexity measures how robustly an implementation handles input shapes and deployment variability.

### Static-Shape Tasks

In static-shape tasks, all input dimensions are known at compile time or graph-construction time. These tasks evaluate specialization and performance optimization. The model is expected to design high-quality tiling strategies, exploit on-chip memory, and minimize unnecessary data movement.

### Dynamic-Shape Tasks

In dynamic-shape tasks, at least some input dimensions are determined at runtime. These tasks evaluate generalization and robustness. The generated code must compute loop bounds, memory offsets, and tiling parameters dynamically, and the host-side implementation must correctly infer output shapes.

## 3. Two-Dimensional Evaluation Matrix

The combination of operator level and shape regime forms the benchmark matrix.

| Operator Level | Static Shape: Optimization Capability | Dynamic Shape: Generalization Capability |
|:--|:--|:--|
| Level 1 | Can the model specialize simple operators and exploit fixed shapes efficiently? | Can the model generate correct general implementations for simple operators? |
| Level 2 | Can the model optimize structured operators with appropriate tiling and reuse? | Can the model preserve correctness across varying input dimensions? |
| Level 3 | Can the model implement complex algorithms efficiently under known structure? | Can the model handle complex control flow and data dependence at runtime? |

## 4. Research Value

This design enables NPUKernelBench to:

- evaluate model capability systematically, from semantic understanding to robust deployment;
- identify failure modes across algorithmic and interface dimensions;
- provide a shared vocabulary for discussing AI-assisted NPU kernel development;
- support future benchmark expansion as LLM-based operator synthesis improves.
