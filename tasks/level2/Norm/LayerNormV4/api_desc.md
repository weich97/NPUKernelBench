# aclnnLayerNormV4

## Functional Description

### Operator Semantics
`aclnnLayerNormV4` is an Ascend NPU benchmark operator in the `level2` `Norm` task family. The implementation should reproduce the reference tensor semantics used by the validation module and expose the custom kernel through `kernel_gen_ops.layer_norm_v4()`.

The task specification is intended for kernel-generation research: candidate implementations should preserve reference-level mathematical behavior while optimizing the device-side execution path for the Ascend C runtime.

### Mathematical Definition
The operator follows the tensor relation below, with shape, dtype, broadcasting, and attribute constraints inherited from the benchmark task configuration when applicable.

$$
out = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + eps}} \cdot w + b
$$

## Interface Definition

### Python Interface
The C++/Ascend implementation is bound to Python through PyBind11 and invoked from the benchmark harness as follows:

```python
def layer_norm_v4(input_tensor, normalized_shape, weight, bias, eps):
    """Execute `aclnnLayerNormV4` on Ascend NPU tensors."""

```

### Inputs
- `input_tensor`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.
- `normalized_shape`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.
- `weight`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.
- `bias`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.
- `eps`: Operator argument supplied by the benchmark input generator. Tensor arguments reside on the device unless the task explicitly defines a host-side scalar or attribute.

### Outputs
- Returns the tensor, tensor list, or in-place updated tensor specified by the reference implementation. Output shape, dtype, layout, and aliasing behavior must be consistent with the validation path.

## Usage Example

```python
import kernel_gen_ops

result = kernel_gen_ops.layer_norm_v4(input_tensor, normalized_shape, weight, bias, eps)
```

## Constraints and Notes

- The implementation must match the PyTorch/reference semantics used in `validation/module.py`.
- Unless otherwise specified by the task configuration, tensors use the `ND` layout and the dtype set declared in the benchmark metadata.
- Candidate kernels should avoid changing public signatures, generated build files, or validation-side calling conventions.
