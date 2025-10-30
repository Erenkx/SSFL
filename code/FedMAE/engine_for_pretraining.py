"""
This module implements the training loop for one epoch for pretraining a
masked autoencoder model.

Originally adapted from the MAE codebase:
https://github.com/facebookresearch/mae
"""

import os
import sys
sys.path.append(os.path.abspath('..'))
import math
from typing import Iterable

import torch

import utils.misc as misc
import utils.lr_sched as lr_sched


def train_one_epoch(
    model: torch.nn.Module,
    global_model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    proxy_single_client=None,
    log_writer=None,
    args=None
):
    model.train()

    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}')
    )
    header = f'Epoch: [{epoch}]'
    print_freq = 20
    accum_iter = args.accum_iter
    optimizer.zero_grad()

    if log_writer is not None:
        print(f'log_dir: {log_writer.log_dir}')

    for data_iter_step, (samples, _) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        args.global_step_per_client[proxy_single_client] += 1

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer,
                data_iter_step / len(data_loader) + epoch,
                args
            )

        samples = samples.to(device, non_blocking=True)

        with torch.amp.autocast('cuda'):
            loss, _, _ = model(samples, mask_ratio=args.mask_ratio)

        loss_value = loss.item()
        metric_logger.update(loss=loss_value)

        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stop training.')
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(
            loss,
            optimizer,
            parameters=model.parameters(),
            update_grad=(data_iter_step + 1) % accum_iter == 0
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        min_lr = 10.0
        max_lr = 0.0
        for param_group in optimizer.param_groups:
            min_lr = min(min_lr, param_group['lr'])
            max_lr = max(max_lr, param_group['lr'])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f'Averaged stats: {metric_logger}')

    if log_writer is not None:
        for k, v in metric_logger.meters.items():
            if k in ['lr']:
                log_writer.add_scalar(
                    proxy_single_client + '/opt/' + k,
                    v.global_avg,
                    log_writer.step
                )
            elif k in ['loss']:
                log_writer.add_scalar(
                    proxy_single_client + '/loss/' + k,
                    v.global_avg,
                    log_writer.step
                )
            
            log_writer.set_step()

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
