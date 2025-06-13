# FeedsRepeat

## 功能描述

### 算子功能
对于输入 `feeds`，根据输入 `feeds_repeat_times`，将对应的 `feeds` 的第 0 维上的数据复制对应的次数，并将输出 `y` 的第 0 维补零到 `output_feeds_size` 的数值。

### 示例
对于 `feeds = {(a, b), (c, d), (e, f)}`，`feeds_repeat_times = {x, y, z}`，则对应在输出里将 `(a, b)` 复制 `x` 次，`(c, d)` 复制 `y` 次，`(e, f)` 复制 `z` 次，若 `output_feeds_size = w + x + y + z`，则在最后再补充 `w` 个对齐的 `(0, 0)`；假设 `feeds_repeat_times = {0, 1, 2}`，`output_feeds_size = 4`，则对应 `out = {(c, d), (e, f), (e, f), (0, 0)}`。

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.feeds_repeat()` 函数形式提供：

```python
def feeds_repeat(feeds, feeds_repeat_times, output_feeds_size):
    """
    实现FeedsRepeat自定义操作。
    
    参数:
        feeds (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        feeds_repeat_times (Tensor): 输入张量，Device侧的张量，数据格式支持ND。
        output_feeds_size (int): 输出的feeds大小
        
    返回:
        Tensor: 计算结果张量，数据类型与feeds一致，数据格式支持ND。
    
    注意:
        张量数据格式支持ND
    """
```

## 使用案例

```python
import torch
import kernel_gen_ops

# 创建输入张量
feeds = torch.randn(2, 3, dtype=torch.float16)
feeds_repeat_times = torch.tensor([100, 200], dtype=torch.int32)
output_feeds_size = 500

# 使用feeds_repeat执行计算
result = kernel_gen_ops.feeds_repeat(feeds, feeds_repeat_times, output_feeds_size)
```

## 约束与限制

- feeds，out的数据格式只支持ND，两者的数据类型一致；
- feeds_repeat_times的数据格式只支持ND，一维tensor，长度（元素个数）必须和feeds的第0维数值相等；
- output_feeds_size为必选属性，数值需大于等于feeds_repeat_times的元素总和；
- 不支持空tensor，feeds_repeat_times的数据规模（Byte大小）不能超过48KB。
    