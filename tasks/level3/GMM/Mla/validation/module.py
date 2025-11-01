from typing import List, Optional

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import torch.nn as nn

import math
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def _group_matmul(self, q, k_or_v):
        """Helper for Grouped-Query Attention matmul."""
        num_heads, q_seqlen, _ = q.shape
        kv_heads = k_or_v.shape[0]

        if num_heads == kv_heads:
            return torch.matmul(q, k_or_v)

        group_num = num_heads // kv_heads

        # Reshape for grouped query attention
        q = q.view(kv_heads, group_num, q_seqlen, -1)
        if k_or_v.dim() == 3: # This is for K
            k = k_or_v.unsqueeze(1)
            score = torch.matmul(q, k)
        else: # This is for V
            v = k_or_v.unsqueeze(1)
            score = torch.matmul(q, v)

        return score.view(num_heads, q_seqlen, -1)

    def _ref_masked_attention(self, query, key, value, scale, mask=None):
        """Performs a single scaled dot-product attention operation."""
        q = query.permute(1, 0, 2)  # (num_heads, q_seqlen, head_size)
        k = key.permute(1, 2, 0)    # (kv_heads, head_size_qk, k_seqlen)
        v = value.permute(1, 0, 2)  # (kv_heads, k_seqlen, head_size)

        scores = self._group_matmul(q, k) * scale
        if mask is not None:
            scores += mask

        attn = F.softmax(scores, dim=-1)
        output = self._group_matmul(attn, v)
        return output.permute(1, 0, 2)

    def forward(self, query_nope, query_rope, kv_nope_cache, kv_rope_cache, block_tables, q_seqlen_list, k_seqlen_list, mask=None):
        """
        Forward pass for the reference Paged Attention model.
        It computes the result in float32 for high precision.
        """
        query = torch.concat([query_nope, query_rope], dim=-1)
        key_cache = torch.concat([kv_nope_cache, kv_rope_cache], dim=-1)
        output_shape = (query.shape[0], query.shape[1], kv_nope_cache.shape[3])
        final_output = torch.empty(output_shape, dtype=torch.float32, device=query_nope.device)
        block_size = kv_nope_cache.shape[1]

        cu_q_seqlen = 0
        for i in range(len(q_seqlen_list)):
            q_len = q_seqlen_list[i]
            k_len = k_seqlen_list[i]

            q_current = query[cu_q_seqlen : cu_q_seqlen + q_len]

            # Reconstruct key and value tensors from the paged cache
            k_list, v_list = [], []
            for j in range(k_len):
                block_idx = j // block_size
                block_offset = j % block_size
                block_number = block_tables[i, block_idx].item()
                k_list.append(key_cache[block_number, block_offset])
                v_list.append(kv_nope_cache[block_number, block_offset])

            keys = torch.stack(k_list, dim=0)
            values = torch.stack(v_list, dim=0)

            scale = 1.0 / (keys.shape[-1] ** 0.5)

            current_mask = mask[cu_q_seqlen:cu_q_seqlen + q_len, :k_len] if mask is not None else None

            # Perform attention, casting to float32 for reference precision
            out = self._ref_masked_attention(
                q_current.to(torch.float32),
                keys.to(torch.float32),
                values.to(torch.float32),
                scale,
                current_mask.to(torch.float32) if current_mask is not None else None
            )
            final_output[cu_q_seqlen : cu_q_seqlen + q_len] = out
            cu_q_seqlen += q_len

        return final_output


class ModelNew(nn.Module):
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(self, query_nope, query_rope, kv_nope_cache, kv_rope_cache, block_tables, q_seqlen_list, k_seqlen_list, mask=None):
        """
        Forward pass that calls the custom operator.
        """
        import kernel_gen_ops
        return kernel_gen_ops.mla(
            query_nope,
            query_rope,
            kv_nope_cache,
            kv_rope_cache,
            block_tables,
            q_seqlen_list,
            k_seqlen_list,
        )