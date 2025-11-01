# prepare_inputs.py
import torch

def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成 CrossEntropyLossGrad 模型的输入张量列表和标量。
    """
    batch_size = param.get('batch', 1)
    num_classes = param.get('num_classes', 1000)
    
    input_dtype_str = param.get('input_dtype', 'float32') # Corresponds to log_prob dtype
    input_dtype = getattr(torch, input_dtype_str)
    
    target_dtype_str = param.get('target_dtype', 'int64')
    target_dtype = getattr(torch, target_dtype_str)

    reduction = param.get('reduction', 'mean')
    if reduction == "none":
        grad_loss = torch.rand((batch_size,), device=device, dtype=input_dtype)
    else: # mean or sum
        grad_loss = torch.rand((), device=device, dtype=input_dtype)
    
    # Using a smaller range for float16/bfloat16 to avoid infinities
    if input_dtype == torch.float16 or input_dtype == torch.bfloat16:
        # Small random values for logits to avoid exp(large_value) overflow
        random_logits = (torch.rand([batch_size, num_classes], device=device, dtype=input_dtype) * 2.0 - 1.0) # Range [-1, 1]
    else: # For float32, a wider range is usually fine
        random_logits = torch.randn([batch_size, num_classes], device=device, dtype=input_dtype) * 2.0 # Standard normal, scaled
    
    log_prob = torch.log_softmax(random_logits, dim=-1)
    # --- END REVISED log_prob generation ---

    target = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device, dtype=target_dtype)

    grad_zloss_type = param.get('grad_zloss_type', 'None')
    grad_zloss = None
    if grad_zloss_type == 'present':
        grad_zloss = torch.rand((1,), device=device, dtype=input_dtype) 

    lse_for_zloss_type = param.get('lse_for_zloss_type', 'None')
    lse_for_zloss = None
    if lse_for_zloss_type == 'present':
        # Generate lse_for_zloss based on original logits to be consistent
        # lse = torch.logsumexp(random_logits, dim=-1) # Calculate LSE from the same random_logits
        # Use rand if you don't want to strictly recompute, but keep it positive
        lse_for_zloss = torch.rand((batch_size,), device=device, dtype=input_dtype) + 1.0 # Ensure positive and not too small
        

    return (grad_loss, log_prob, target, grad_zloss, lse_for_zloss)


def get_init_inputs(param, device=None):
    num_classes = param.get('num_classes', 1000)
    
    weight_type = param.get('weight_type', 'present')
    weight_dtype_str = param.get('weight_dtype', 'float32')
    weight_dtype = getattr(torch, weight_dtype_str)

    ignore_index = param.get('ignore_index', -100)
    label_smoothing = float(param.get('label_smoothing', 0.0))
    reduction = param.get('reduction', 'mean')
    lse_square_scale_for_zloss = float(param.get('lse_square_scale_for_zloss', 0.0))
    
    weight = None
    if weight_type == 'present':
        weight = torch.rand(num_classes, device=device, dtype=weight_dtype)
    
    return [weight, ignore_index, label_smoothing, reduction, lse_square_scale_for_zloss]

def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get('input_dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float16:
        RTOL_GENERAL = 1 / 512
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 256 + 1 / 16384
    elif dtype == torch.float32:
        RTOL_GENERAL = 1 / 2048 + 1 / 16384

    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1
    for out, out_new in zip(outputs, outputs_new):
        # 计算绝对差值、相对误差
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        all_abs_diff.append(abs_diff.view(-1))
        all_rel_diff.append(rel_diff.view(-1))

        # 计算容忍度阈值
        tolerance = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out))

        # 找出差异大于容忍度的位置
        error_mask = abs_diff > tolerance

        # 检查是否有任何元素的差异超过了容忍度
        if torch.any(error_mask):
            is_pass = 0

    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)

    return is_pass, all_abs_diff, all_rel_diff