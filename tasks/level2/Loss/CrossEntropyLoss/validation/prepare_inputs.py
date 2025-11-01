# prepare_inputs.py
import torch

def get_inputs(param, device=None):
    """
    Generate input tensors for the CrossEntropyLoss operator's forward method.
    """
    batch_size = param.get('batch', 1)
    num_classes = param.get('num_classes', 1000)
    
    input_dtype_str = param.get('input_dtype', 'float16')
    input_dtype = getattr(torch, input_dtype_str)
    
    target_dtype_str = param.get('target_dtype', 'int64')
    target_dtype = getattr(torch, target_dtype_str)

    input_predictions = torch.randn([batch_size, num_classes], device=device, dtype=input_dtype)
    target_labels = torch.randint(low=0, high=num_classes, size=(batch_size,), device=device, dtype=target_dtype)

    return (input_predictions, target_labels)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the CrossEntropyLoss model.
    """
    num_classes = param.get('num_classes', 1000)
    
    weight_type = param.get('weight_type', 'present')
    weight_dtype_str = param.get('weight_dtype', 'float32') # Default weight dtype
    weight_dtype = getattr(torch, weight_dtype_str)

    ignore_index = param.get('ignore_index', -100)
    label_smoothing = float(param.get('label_smoothing', 0.0)) # Ensure float
    reduction = param.get('reduction', 'mean') # String
    lse_square_scale_for_zloss = float(param.get('lse_square_scale_for_zloss', 0.0)) # Ensure float
    
    # --- START CHANGE ---
    # Properly parse return_zloss from CSV
    return_zloss_str = str(param.get('return_zloss', False)).lower()
    return_zloss = (return_zloss_str == 'true' or return_zloss_str == '1')
    # --- END CHANGE ---

    weight = None
    if weight_type == 'present':
        weight = torch.rand(num_classes, device=device, dtype=weight_dtype)
    
    return [weight, ignore_index, label_smoothing, reduction, lse_square_scale_for_zloss, return_zloss]