# GroupedMatmul

## 功能描述

### 算子功能
`GroupedMatmul` 实现了分组矩阵乘法计算，支持多组不同维度的矩阵乘法操作在单次算子调用中完成。该算子特别适用于需要并行执行多个独立矩阵乘法的场景，如多头注意力机制、分组卷积等。

### 计算公式

基本的分组矩阵乘法计算公式为：

$$y_i = x_i \times weight_i + bias_i, \quad i=1,2,...,g$$

其中：
- $g$ 是分组个数
- $x_i \in \mathbb{R}^{m_i \times k_i}$ 是第 $i$ 组的输入矩阵
- $weight_i \in \mathbb{R}^{k_i \times n_i}$ 是第 $i$ 组的权重矩阵
- $bias_i \in \mathbb{R}^{n_i}$ 是第 $i$ 组的偏置向量（可选）
- $y_i \in \mathbb{R}^{m_i \times n_i}$ 是第 $i$ 组的输出矩阵
- $m_i$、$k_i$、$n_i$ 分别表示第 $i$ 组矩阵的相应维度

当启用转置时：
- 若 `transpose_x=True`：$x_i^T$ 的形状为 $k_i \times m_i$
- 若 `transpose_weight=True`：$weight_i^T$ 的形状为 $n_i \times k_i$

## 接口定义

### Python 接口
该操作通过 PyBind11 封装 C++ 实现，在 Python 中以 `kernel_gen_ops.grouped_matmul()` 函数形式提供：

```python
def grouped_matmul(x, weight, bias=None, group_list=None, split_item=0, 
                   transpose_weight=False, transpose_x=False):
    """
    实现分组矩阵乘法操作。
    
    参数:
        x (Tensor or List[Tensor]): 输入张量或张量列表。
            - 当为单个Tensor时，需配合group_list参数指定分组方式
            - 当为List[Tensor]时，每个Tensor对应一组独立的输入
            支持的数据类型：FLOAT16、BFLOAT16、FLOAT32
            数据格式支持ND，维度范围2-6维
            
        weight (List[Tensor]): 权重张量列表，每个Tensor对应一组权重矩阵。
            必须为2维张量，数据类型需与x保持一致
            
        bias (List[Tensor], optional): 偏置张量列表，默认为None。
            若提供，每个Tensor为1维向量，长度需与对应weight的列数相同
            
        group_list (List[int], optional): 分组索引列表，默认为None。
            仅在x为单Tensor时有效，指定x在M轴上的分组边界
            例如：[50, 65, 84]表示x[0:50]、x[50:65]、x[65:84]分别参与三组计算
            
        split_item (int): 输出格式控制标志，默认为0。
            - 0或1：输出为List[Tensor]，每组结果独立存储
            - 2或3：输出为单个Tensor，所有结果在M轴上拼接
            
        transpose_weight (bool): 是否转置权重矩阵，默认为False。
            
        transpose_x (bool): 是否转置输入矩阵，默认为False。
        
    返回:
        Tensor or List[Tensor]: 计算结果。
        - 当split_item为0或1时，返回张量列表，每个张量对应一组输出
        - 当split_item为2或3时，返回单个张量，所有输出在M轴上拼接
        输出数据类型与输入保持一致
    """
```

## 使用案例

### 案例1：多多多场景 - 多头注意力机制
```python
import torch
import kernel_gen_ops

# 模拟多头注意力中的多个独立矩阵乘法
# 3个注意力头，每个头的维度可能不同
x_list = [
    torch.randn(32, 128, dtype=torch.float16),   # 头1：batch_size=32, hidden_dim=128
    torch.randn(32, 256, dtype=torch.float16),   # 头2：batch_size=32, hidden_dim=256
    torch.randn(32, 512, dtype=torch.float16)    # 头3：batch_size=32, hidden_dim=512
]

weight_list = [
    torch.randn(128, 64, dtype=torch.float16),   # 头1：投影到64维
    torch.randn(256, 64, dtype=torch.float16),   # 头2：投影到64维
    torch.randn(512, 64, dtype=torch.float16)    # 头3：投影到64维
]

bias_list = [
    torch.randn(64, dtype=torch.float16),
    torch.randn(64, dtype=torch.float16),
    torch.randn(64, dtype=torch.float16)
]

# 执行分组矩阵乘法
outputs = kernel_gen_ops.grouped_matmul(x_list, weight_list, bias_list, split_item=0)

# outputs是一个列表，包含3个tensor，每个形状为[32, 64]
for i, output in enumerate(outputs):
    print(f"Head {i+1} output shape: {output.shape}")
```

### 案例2：单多单场景 - 动态批处理
```python
import torch
import kernel_gen_ops

# 场景：不同长度的序列拼接在一起，需要分别处理
# 假设有4个不同长度的序列批次
batch_sizes = [50, 30, 40, 30]
total_size = sum(batch_sizes)
hidden_dim = 512
output_dim = 256

# 所有序列拼接成一个大tensor
x = torch.randn(total_size, hidden_dim, dtype=torch.float16)

# 每个批次使用不同的权重（但输出维度相同）
weight_list = [
    torch.randn(hidden_dim, output_dim, dtype=torch.float16) for _ in range(4)
]

bias_list = [
    torch.randn(output_dim, dtype=torch.float16) for _ in range(4)
]

# 定义分组边界（累积和）
group_list = []
cumsum = 0
for size in batch_sizes:
    cumsum += size
    group_list.append(cumsum)
# group_list = [50, 80, 120, 150]

# 执行分组矩阵乘法，输出合并为单个tensor
result = kernel_gen_ops.grouped_matmul(
    x, weight_list, bias_list,
    group_list=group_list, 
    split_item=2
)

print(f"Merged output shape: {result.shape}")  # [150, 256]
```

### 案例3：单多多场景 - 分组线性层
```python
import torch
import kernel_gen_ops

# 场景：共享输入特征，但需要生成多个不同的输出
# 例如：多任务学习中的共享编码器
batch_size = 64
shared_features = 1024

# 共享的输入特征
x = torch.randn(batch_size, shared_features, dtype=torch.float16)

# 不同任务的投影矩阵
weight_list = [
    torch.randn(shared_features, 128, dtype=torch.float16),  # 任务1：分类（128类）
    torch.randn(shared_features, 256, dtype=torch.float16),  # 任务2：回归（256维）
    torch.randn(shared_features, 64, dtype=torch.float16),   # 任务3：特征提取（64维）
]

# 每个批次的所有样本都参与所有任务
group_list = [batch_size, batch_size, batch_size]  # 每组都是完整的batch

# 执行分组矩阵乘法，每个任务输出独立
outputs = kernel_gen_ops.grouped_matmul(
    x, weight_list,
    group_list=group_list,
    split_item=0
)

# outputs包含3个tensor，分别用于不同任务
print(f"Task 1 output: {outputs[0].shape}")  # [64, 128]
print(f"Task 2 output: {outputs[1].shape}")  # [64, 256]
print(f"Task 3 output: {outputs[2].shape}")  # [64, 64]
```

### 案例4：多多单场景 - 特征融合
```python
import torch
import kernel_gen_ops

# 场景：多个不同来源的特征需要融合到统一维度
# 例如：多模态学习中的特征融合

# 不同模态的特征
x_list = [
    torch.randn(32, 512, dtype=torch.float16),   # 图像特征
    torch.randn(48, 768, dtype=torch.float16),   # 文本特征
    torch.randn(24, 256, dtype=torch.float16),   # 音频特征
]

# 投影到相同的维度进行融合
fusion_dim = 128
weight_list = [
    torch.randn(512, fusion_dim, dtype=torch.float16),  # 图像投影
    torch.randn(768, fusion_dim, dtype=torch.float16),  # 文本投影
    torch.randn(256, fusion_dim, dtype=torch.float16),  # 音频投影
]

group_list = [32, 80, 104]

# 执行分组矩阵乘法，输出合并存储
fused_features = kernel_gen_ops.grouped_matmul(
    x_list, weight_list,
    split_item=3
)


print(f"Fused features shape: {fused_features.shape}")  # [104, 128]
# 其中 104 = 32 + 48 + 24
```

## 约束与限制

### 基本约束

如果传入group_list，group_list必须为非负递增数列，group_list长度不能为1。

### 场景支持约束

当前支持的场景：
支持场景中单表示单tensor，多表示多tensor，表示顺序为x，weight，y，例如，单多单表示支持x为单tensor，weight多tensor，y单tensor的场景。

| 支持场景 | 场景限制 |
|---------|---------|
| 多多多 | 1）仅支持split_item为0/1<br>2）x中tensor支持2-6维，weight中tensor需为2维，y中tensor维度和x保持一致<br>3）若x中存在tensor大于2维，group_list必须传空<br>4）若x中tensor为2维且传入group_list，group_list的差值需与x中tensor的第一维一一对应 |
| 单多单 | 1）仅支持split_item为2/3<br>2）必须传group_list，且最后一个值与x中tensor的第一维相等<br>3）x,weight,y中tensor需为2维<br>4）weight中每个tensor的N轴必须相等 |
| 单多多 | 1）仅支持split_item为0/1<br>2）必须传group_list，group_list的差值需与y中tensor的第一维一一对应<br>3）x,weight,y中tensor需为2维 |
| 多多单 | 1）仅支持split_item为2/3<br>2）x,weight,y中tensor需为2维<br>3）weight中每个tensor的N轴必须相等<br>4）若传入group_list，group_list的差值需与x中tensor的第一维一一对应 |

### 维度大小约束

x和weight中每一组tensor的最后一维大小都应小于65536。$x_i$的最后一维指当属性transpose_x为false时$x_i$的K轴或当transpose_x为true时$x_i$的M轴。$weight_i$的最后一维指当属性transpose_weight为false时$weight_i$的N轴或当transpose_weight为true时$weight_i$的K轴。

x和weight中每一组tensor的每一维大小在32字节对齐后都应小于int32的最大值2147483647。

### 数据类型约束

- x、weight、bias（如果提供）的数据类型必须一致
- 支持的数据类型：FLOAT16、BFLOAT16、FLOAT32
- 不支持整数类型和量化类型

### 数量约束

- weight列表的最大长度为128
- 当提供bias时，bias列表长度必须与weight列表长度相同
- 当x为单tensor且提供group_list时，group_list长度必须与weight列表长度相同

### 其他约束

- 所有输入输出张量的数据格式支持ND
- 支持非连续的Tensor
- 当bias不为空时，每个bias必须为1维张量，且长度与对应weight的N轴（列数）相同