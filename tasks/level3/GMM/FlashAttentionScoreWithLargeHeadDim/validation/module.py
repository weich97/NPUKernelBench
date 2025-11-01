from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn
import torch.nn.functional as F
import kernel_gen_ops

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def forward(self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_value: float,
        head_num: int,
    ) -> torch.Tensor:  
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        value = value.to(torch.float32)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores * scale_value
        attention_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, value)
        return output.to(torch.float16)

class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, 
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        scale_value: float,
        head_num: int,
    ) -> torch.Tensor: 
        output = kernel_gen_ops.flash_attention_score_with_large_head_dim(query, key, value, scale_value, head_num)
        return output