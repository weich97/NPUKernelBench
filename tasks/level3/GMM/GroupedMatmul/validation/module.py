from typing import List, Optional, Union

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn


class Model(nn.Module):
    """
    实现grouped matmul算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]],
                weight: List[torch.Tensor],
                bias: Optional[List[torch.Tensor]] = None,
                group_list: Optional[List[int]] = None,
                split_item: int = 0,
                transpose_weight: bool = False,
                transpose_x: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        实现grouped matmul算子功能。

        Args:
            x: 输入张量或张量列表
            weight: 权重张量列表
            bias: 偏置张量列表（可选）
            group_list: 分组列表（可选）
            split_item: 输出分割标志
            transpose_weight: 是否转置权重
            transpose_x: 是否转置输入

        Returns:
            输出张量或张量列表
        """
        outputs = []

        # 处理输入x
        if isinstance(x, torch.Tensor):
            # 单tensor情况，需要根据group_list分割
            if group_list is not None:
                x_list = []
                start = 0
                for end in group_list:
                    x_list.append(x[start:end])
                    start = end
            else:
                x_list = [x]
        else:
            x_list = x

        # 执行分组矩阵乘法
        for i, (x_i, w_i) in enumerate(zip(x_list, weight)):
            # 处理转置
            if transpose_x:
                x_i = x_i.t()
            if transpose_weight:
                w_i = w_i.t()

            # 矩阵乘法
            y_i = torch.matmul(x_i, w_i)

            # 添加偏置
            if bias is not None:
                y_i = y_i + bias[i]

            outputs.append(y_i)

        # 根据split_item决定输出格式
        if split_item in [2, 3]:
            # 合并输出为单tensor
            return torch.cat(outputs, dim=0)
        else:
            # 返回tensor列表
            return outputs


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor]],
                weight: List[torch.Tensor],
                bias: Optional[List[torch.Tensor]] = None,
                group_list: Optional[List[int]] = None,
                split_item: int = 0,
                transpose_weight: bool = False,
                transpose_x: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        import kernel_gen_ops
        return kernel_gen_ops.grouped_matmul(x, weight, bias, group_list, split_item, transpose_weight, transpose_x)