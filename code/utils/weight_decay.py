"""
This module provides a utility function to add weight decay to model parameters.

Originally adapted from the SSL-FL codebase:
https://github.com/rui-yan/SSL-FL
"""


def add_weight_decay(
    model, 
    weight_decay=1e-5, 
    lora_weight_decay=0.01, 
    skip_list=()):
    """
    Adds weight decay to the given model's parameters.

    Args:
        model (torch.nn.Module): The model to add weight decay to.
        weight_decay (float): The weight decay coefficient. Default is 
            1e-5.
        skip_list (list): A list of parameter names to skip weight 
            decay. Default is ().

    Returns:
        list: A list of parameter groups with and without weight decay.
    """
    decay = []
    no_decay = []
    lora_decay = []

    for name, param in model.named_parameters():
        # Skip frozen parameters
        if not param.requires_grad:
            continue

        if (
            len(param.shape) == 1 
            or name.endswith('.bias') 
            or name in skip_list
        ):
            no_decay.append(param)
        elif '.lora_A' in name or '.lora_B' in name:
            lora_decay.append(param)
        else:
            decay.append(param)
    
    return [
        {'params': no_decay, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': lora_decay, 'weight_decay': lora_weight_decay}
    ]
