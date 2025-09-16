"""
This module provides a learning rate scheduler with cosine decay.

Originally adapted from the MAE codebase:
https://github.com/facebookresearch/mae
"""

import math


def adjust_learning_rate(optimizer, epoch, args):
    """
    Applies cosine decay learning rate with linear warmup.
    """
    if epoch < args.warmup_epochs: # Linear warmup
        lr = args.lr * epoch / args.warmup_epochs
    else: # Cosine decay
        lr = args.lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(
                math.pi * (epoch - args.warmup_epochs)
                / (args.max_communication_rounds - args.warmup_epochs)
            )
        )

    for param_group in optimizer.param_groups:
        if 'lr_scale' in param_group:
            param_group['lr'] = lr * param_group['lr_scale']
        else:
            param_group['lr'] = lr

    return lr
