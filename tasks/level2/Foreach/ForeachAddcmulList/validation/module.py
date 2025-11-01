from typing import List

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import kernel_gen_ops


class Model(nn.Module):
    """
    实现ForeachAddcmulList算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(
        self,
        x1: List[torch.Tensor],
        x2: List[torch.Tensor],
        x3: List[torch.Tensor],
        scalars: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        实现 ForeachAddcmulList 算子功能。

        步骤：
        1. 先把 x1、x2、x3 中的每个张量提升到 float32；
        2. scalars 保持原 dtype 不变；
        3. 执行计算：x + scalar * y * z；
        4. 返回 float32 结果列表。
        """
        ret = []
        for scalar, x, y, z in zip(scalars, x1, x2, x3):
            if x.dtype == torch.bfloat16:
                ret.append(x.to(torch.float32) + scalar.to(torch.float32) * y.to(torch.float32) * z.to(torch.float32))
            else:
                ret.append(x + scalar * y * z)
        return ret

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x1: List[torch.Tensor], x2: List[torch.Tensor], x3: List[torch.Tensor], scalars: torch.Tensor) -> List[torch.Tensor]:
        return kernel_gen_ops.foreach_addcmul_list(x1, x2, x3, scalars)