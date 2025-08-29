"""
Learning rate decay utilities.

Originally adapted from the MAE codebase:
https://github.com/facebookresearch/mae
"""

def get_layer_id_for_vit(name, num_layers):
    """
    Assigns a parameter to a layer ID for ViT models.

    Reference:
        https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L33
    """
    if name in ['cls_token', 'pos_embed']:
        return 0
    elif name.startswith('patch_embed'):
        return 0
    elif name.startswith('blocks'):
        return int(name.split('.')[1]) + 1
    else:
        return num_layers


def param_groups_lrd(
    model,
    weight_decay=0.05,
    no_weight_decay_list=[],
    layer_decay=0.75
):
    """
    Parameter groups for layer-wise learning rate decay.

    Reference:
        https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}

    num_layers = len(model.blocks) + 1
    layer_scales = [
        layer_decay ** (num_layers - i) for i in range(num_layers + 1)
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if param.ndim == 1 or name in no_weight_decay_list:
            group_type = 'no_decay'
            decay = 0.0
        else:
            group_type = 'decay'
            decay = weight_decay

        layer_id = get_layer_id_for_vit(name, num_layers)
        group_name = f'layer_{layer_id}_{group_type}'

        if group_name not in param_group_names:
            scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                'lr_scale': scale,
                'weight_decay': decay,
                'params': []
            }
            param_groups[group_name] = {
                'lr_scale': scale,
                'weight_decay': decay,
                'params': []
            }

        param_group_names[group_name]['params'].append(name)
        param_groups[group_name]['params'].append(param)

    return list(param_groups.values())
