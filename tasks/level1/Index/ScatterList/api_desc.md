# aclnnScatterList

## 功能描述

### 算子功能

该Ascend C算子用于根据指定的索引（`indice`）和轴（`axis`），将更新张量（`updates`）中的数据更新到目标张量列表（`var`）中。此操作是in-place的，即会直接修改`var`中的数据。此外，可以通过一个可选的掩码（`mask`）来决定是否对`var`中的某个张量执行更新操作。

### 计算公式

该算子的核心功能是切片和赋值。对于`var`列表中的第`i`个张量`var[i]`，如果`mask[i]`为`True`，则执行以下操作：

1.  根据`indice[i]`的值确定目标切片和源切片。`indice[i]`是一个包含两个元素`[start, length]`的张量，其中`start`是切片的起始位置，`length`是切片的长度。
2.  在指定的`axis`上，确定目标张量`var[i]`的切片范围为`dest_slice = slice(start, start + length)`。
3.  在指定的`axis`上，确定更新张量`updates[i]`的切片范围为`source_slice = slice(0, length)`。
4.  将`updates[i]`中由`source_slice`切出的数据块，赋值给`var[i]`中由`dest_slice`指定的位置。

伪代码如下：

```python
for i in range(len(var)):
  if mask is not None and mask[i] == False:
    continue
  
  start = indice[i][0]
  length = indice[i][1]
  
  dest_slice = [slice(None)] * var[i].ndim
  dest_slice[axis] = slice(start, start + length)
  
  source_slice = [slice(None)] * updates[i].ndim
  source_slice[axis] = slice(0, length)
  
  var[i][tuple(dest_slice)] = updates[i][tuple(source_slice)]
```

## 接口定义

### 算子原型定义接口

#### Input

  - **var**: Device侧的`aclTensor`列表，是需要被更新的目标张量列表。`var`列表中的所有张量必须具有相同的形状和数据类型。
      - 数据类型支持: `int8`, `int16`, `int32`, `uint8`, `uint16`, `uint32`, `float16`, `bfloat16`, `float32`。
      - 数据格式支持: `ND`。
  - **indice**: Device侧的`aclTensor`，用于指定更新位置的索引。
      - 数据类型支持: `int32`, `int64`，默认为`int64`。
      - 数据格式支持: `ND`。
  - **updates**: Device侧的`aclTensor`，包含用于更新的数据。
      - 数据类型必须与`var`中的张量相同。
      - 数据格式支持: `ND`。
  - **mask**: Device侧可选的`aclTensor`，用于决定是否对`var`中的某个张量执行更新。
      - 数据类型支持: `uint8`。
      - 数据格式支持: `ND`。

#### Output

  - **var**: Device侧的`aclTensor`列表，返回更新后的`var`张量列表。该操作是in-place的，所以输出与输入是同一个对象。

#### Attr

  - **reduce**: `string`类型，可选属性，当前只支持`"update"`。
  - **axis**: `int`类型，可选属性，指定执行更新操作的轴，默认为`-2`。

## 约束与限制

  * `var`中的所有张量必须具有相同的形状。
  * `var`中的张量数量必须与`updates`的第一个维度的大小相同。
  * `indice`的第一个维度的大小必须与`updates`的第一个维度的大小相同。
  * 如果提供了`mask`，`mask`的第一个维度的大小必须与`updates`的第一个维度的大小相同。
  * `indice`的维度必须为1或2。如果维度为2，则第二个维度的大小必须为2。
  * `updates`张量在`axis`指定的维度上的大小，不能超过`var`中张量在`axis`指定的维度上的大小。
  * 除了`axis`指定的维度以及第一个维度外，`var`和`updates`的其他维度大小必须相等。