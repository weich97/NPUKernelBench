import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for the Paged Attention model based on parameters.

    Args:
        param (dict): Parameters from a pandas DataFrame row.
        device (str, optional): The device to place tensors on. Defaults to None.

    Returns:
        tuple: A tuple of input tensors (query, key_cache, value_cache,
               block_tables, q_seqlen_list, k_seqlen_list, mask).
    """
    # Extract parameters from the dictionary
    batch_size = int(param.get('batch'))
    q_seqlen = int(param.get('q_seqlen'))
    kv_seqlen = int(param.get('kv_seqlen'))
    num_heads = int(param.get('num_heads'))
    kv_heads = int(param.get('kv_heads'))
    head_size = int(param.get('head_size'))
    head_size_rope = int(param.get('head_size_rope'))
    num_blocks = int(param.get('num_blocks'))
    block_size = int(param.get('block_size'))
    mask_type = 0
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    assert batch_size * kv_seqlen <= num_blocks * block_size,  "[ERROR] the number of K and V tokens is too big to fit in the paged cache."

    assert block_size == 128, "[ERROR] blockSize != 128 is not supported."

    assert q_seqlen <= 4, "[ERROR] q_seqlen > 4 is not supported."

    # Create sequence length lists for each item in the batch
    q_seqlen_list = [q_seqlen] * batch_size
    k_seqlen_list = [kv_seqlen] * batch_size

    num_tokens = sum(q_seqlen_list)
    head_size_qk = head_size + head_size_rope

    # Generate random tensors for query and KV cache
    query = torch.rand(num_tokens, num_heads, head_size_qk, device=device, dtype=dtype) * 2 - 1
    query_nope = query[:, :, :head_size]
    query_rope = query[:, :, -head_size_rope:]
    key_cache = torch.rand(num_blocks, block_size, kv_heads, head_size_qk, device=device, dtype=dtype) * 2 - 1
    kv_nope_cache = key_cache[:, :, :, :head_size]
    kv_rope_cache = key_cache[:, :, :, -head_size_rope:]

    # Generate block tables to map tokens to cache blocks
    max_k_seqlen = max(k_seqlen_list)
    max_num_blocks_per_seq = (max_k_seqlen + block_size - 1) // block_size

    block_tables_list = []
    # Allocate non-overlapping blocks for each sequence in the batch for simplicity
    for i in range(batch_size):
        block_table = torch.arange(
            start=i * max_num_blocks_per_seq,
            end=(i + 1) * max_num_blocks_per_seq,
            dtype=torch.int32,
            device=device
        )
        block_tables_list.append(block_table)
    block_tables = torch.stack(block_tables_list)

    # Generate attention mask if specified
    mask = None
    if mask_type == 1:  # Indicates a causal mask
        pre_mask_factor = -10000.0
        mask = torch.zeros(num_tokens, max_k_seqlen, device=device, dtype=dtype)

        pre_qseqlen = 0
        for i in range(batch_size):
            qlen = q_seqlen_list[i]
            klen = k_seqlen_list[i]

            # Create an upper-triangular matrix for the causal mask
            causal_mask = torch.triu(torch.ones(qlen, qlen, device=device, dtype=dtype), diagonal=1) * pre_mask_factor

            if klen >= qlen:
                mask[pre_qseqlen:(pre_qseqlen + qlen), klen - qlen:klen] = causal_mask

            pre_qseqlen += qlen

    return (query_nope, query_rope, kv_nope_cache, kv_rope_cache, block_tables, q_seqlen_list, k_seqlen_list, mask)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for sinh.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    return []  # No special initialization inputs needed

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float16:
        RTOL_GENERAL = 1 / 256
        RTOL_OVER_THRESHOLD = 1 / 128
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 128
        RTOL_OVER_THRESHOLD = 1 / 64
    elif dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048
        RTOL_OVER_THRESHOLD = 1 / 512

    compute_num = int(param.get('head_size'))
    COMPUTE_NUM_THRESHOLD = 2048

    outputs = outputs.to(torch.float32)
    outputs_new = outputs_new.to(torch.float32)

    # 根据 compute_num 的值选择相对容忍度 (rtol)
    rtol = RTOL_GENERAL if compute_num < COMPUTE_NUM_THRESHOLD else RTOL_OVER_THRESHOLD

    # 1. 计算绝对差值
    abs_diff = torch.abs(outputs - outputs_new)

    # 2. 计算容忍度阈值
    tolerance = rtol * torch.maximum(torch.tensor(1.0, device=outputs.device), torch.abs(outputs))

    # 3. 找出差异大于容忍度的位置
    error_mask = abs_diff > tolerance

    # 检查是否有任何元素的差异超过了容忍度
    is_pass = 1 if not torch.any(error_mask) else 0

    # 计算相对误差时，为防止分母为零，加上一个极小值 epsilon
    relative_diff = abs_diff / (torch.abs(outputs) + 1e-7)

    return is_pass, abs_diff, relative_diff