from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math


class Model(nn.Module):
    """
    实现FeedsRepeat算子功能的模型。
    """

    def __init__(self):
        """
        初始化模型。
        """
        super(Model, self).__init__()

    def forward(self, feeds: torch.Tensor, feeds_repeat_times: torch.Tensor, output_feeds_size: int) -> torch.Tensor:
        """
        实现FeedsRepeat算子功能。

        Args:
            feeds: 输入张量
            feeds_repeat_times: 重复次数张量
            output_feeds_size: 输出的feeds大小

        Returns:
            处理后的输出张量
        """

        # 重复张量元素
        repeated = torch.repeat_interleave(feeds, feeds_repeat_times, dim=0)

        # 计算需要填充的数量
        total_repeated = feeds_repeat_times.sum().item()
        pad_size = output_feeds_size - total_repeated

        # 如果需要填充
        if pad_size > 0:
            # 创建输出张量
            output_shape = (output_feeds_size,) + feeds.shape[1:]
            output = torch.zeros(output_shape,
                                 dtype=feeds.dtype,
                                 device=feeds.device)

            # 将重复后的数据拷贝到输出中
            output[:total_repeated] = repeated
            return output
        else:
            # 如果不需要填充，直接返回重复后的数据
            return repeated


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, feeds: torch.Tensor, feeds_repeat_times: torch.Tensor, output_feeds_size: int) -> torch.Tensor:
        import kernel_gen_ops
        return kernel_gen_ops.feeds_repeat(feeds, feeds_repeat_times, output_feeds_size)
