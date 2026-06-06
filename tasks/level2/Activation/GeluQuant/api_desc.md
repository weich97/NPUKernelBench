# GeluQuant

## Functional Description

### Operator Semantics
`GeluQuant` is an Ascend NPU benchmark operator in the `level2` `Activation` task family. The implementation should reproduce the reference tensor semantics used by the validation module and expose the custom kernel through `kernel_gen_ops.gelu_quant()`.

The task specification is intended for kernel-generation research: candidate implementations should preserve reference-level mathematical behavior while optimizing the device-side execution path for the Ascend C runtime.

### Mathematical Definition
The operator follows the tensor relation below, with shape, dtype, broadcasting, and attribute constraints inherited from the benchmark task configuration when applicable.

$$
   \text{gelu} = \text{GELU}(x,\ \text{approximate})
   $$

$$
   y_{\text{out}} = \Bigl\lfloor \text{gelu} \cdot \text{inputScale} + \text{inputOffset} \Bigr\rceil
   \ \text{clip}_{[-128,\,127]}
   $$

$$
   \begin{aligned}
   \text{temp} &= \text{gelu} \cdot \text{inputScale} \\
   \text{maxAbs} &= \max(|\text{temp}|) \\[2pt]
   \text{outScaleOut} &= \frac{\text{maxAbs}}{127.0} \\[4pt]
   y_{\text{out}} &= \Bigl\lfloor \frac{\text{temp}}{\text{outScaleOut}} \Bigr\rceil
   \ \text{clip}_{[-128,\,127]}
   \end{aligned}
   $$

## Interface Definition

### Python Interface
The C++/Ascend implementation is bound to Python through PyBind11 and invoked from the benchmark harness as follows:

```python
def gelu_quant(*args, **kwargs):
    """Execute `GeluQuant` on Ascend NPU tensors."""

```

### Inputs
- Operator arguments are supplied by the benchmark input generator and follow the reference validation signature.

### Outputs
- Returns the tensor, tensor list, or in-place updated tensor specified by the reference implementation. Output shape, dtype, layout, and aliasing behavior must be consistent with the validation path.

## Usage Example

```python
import kernel_gen_ops

result = kernel_gen_ops.gelu_quant(...)
```

## Constraints and Notes

- The implementation must match the PyTorch/reference semantics used in `validation/module.py`.
- Unless otherwise specified by the task configuration, tensors use the `ND` layout and the dtype set declared in the benchmark metadata.
- Candidate kernels should avoid changing public signatures, generated build files, or validation-side calling conventions.
