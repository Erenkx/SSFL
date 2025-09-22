"""
This module provides miscellaneous utilities for distributed training, logging, and checkpointing.

Originally adapted from the SSL-FL codebase:
https://github.com/rui-yan/SSL-FL
"""

import io
import os
import math
import time
import datetime
from pathlib import Path
from collections import defaultdict, deque

import torch
import numpy as np
import torch.nn as nn
import torch.distributed as dist
from timm.utils import get_state_dict


def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def get_world_size():
    return dist.get_world_size() if is_dist_avail_and_initialized() else 1


def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def fix_random_seeds(args):
    """
    Fixes random seeds for reproducibility across Torch, CUDA, and 
    NumPy.
    """
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Loads EMA weights from an already-loaded checkpoint object.
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def setup_for_distributed(is_master):
    """
    Disables printing on non-master processes to avoid redundant logs.
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*arg, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*arg, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    """
    Initializes distributed training mode based on the environment 
    variables.
    """
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        
        master_addr = os.environ['MASTER_ADDR']
        master_port = os.environ['MASTER_PORT']
        args.dist_url = f'tcp://{master_addr}:{master_port}'
        
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False

        return
    
    args.distributed = True
    args.dist_backend = 'nccl'

    torch.cuda.set_device(args.gpu)
    print(
        f'| distributed init (rank {args.rank}): {args.dist_url}, '
        f'gpu {args.gpu} |', flush=True
    )

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def load_state_dict(
    model, 
    state_dict, 
    prefix='',
    ignore_missing='relative_position_index'
):
    """
    Loads a state_dict into the model with support for ignoring specific
    missing keys.
    
    Args:
        model (nn.Module): The model to load the weights into.
        state_dict (dict): The state dictionary from a checkpoint.
        prefix (str): Optional prefix to add when loading the keys. 
            Default is ''.
        ignore_missing (str): Pipe-separated list of substrings to 
            ignore in missing keys. 
            Default is 'relative_position_index'.
    """
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    # Copy state_dict and metadata
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = (
            {} 
            if metadata is None
            else metadata.get(prefix[:-1], {})
        )
        module._load_from_state_dict(
            state_dict, prefix, local_metadata, 
            strict=True,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs
        )

        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix)

    # Filter ignored keys
    warn_missing_keys = []
    ignore_missing_keys = []
    for key in missing_keys:
        if any(ignore in key for ignore in ignore_missing.split('|')):
            ignore_missing_keys.append(key)
        else:
            warn_missing_keys.append(key)

    # Logging
    if warn_missing_keys:
        print(
            f'Weights of {model.__class__.__name__} not initialized from '
            f'pretrained model:\n{warn_missing_keys}'
        )

    if unexpected_keys:
        print(
            f'Weights from pretrained model not used in '
            f'{model.__class__.__name__}:\n{unexpected_keys}'
        )
        
    if ignore_missing_keys:
        print(
            'Ignored weights not initialized from pretrained model:\n'
            f'{ignore_missing_keys}'
        )

    if error_msgs:
        print(
            'Errors occurred while loading state dict:\n' +
            '\n'.join(error_msgs)
        )


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    """
    Computes the gradient norm of a model's parameters.

    Args:
        parameters (iterable): Model parameters with `.grad` attributes.
        norm_type (float): Type of norm to use (e.g., 2.0 for L2, 
            inf for max).

    Returns:
        torch.Tensor: The total gradient norm.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    parameters = [p for p in parameters if p.grad is not None]

    if not parameters:
        return torch.tensor(0.0)
    
    device = parameters[0].grad.device
    norm_type = float(norm_type)

    if norm_type == math.inf:
        total_norm = max(
            p.grad.detach().abs().max().to(device) for p in parameters
        )
    else:
        total_norm = torch.norm(
            torch.stack([
                torch.norm(p.grad.detach(), norm_type).to(device)
                for p in parameters
            ]),
            norm_type
        )

    return total_norm


def save_model(
    args,
    epoch,
    model,
    model_without_ddp,
    optimizer,
    loss_scaler,
    model_ema=None
):
    """
    Saves model checkpoint for the current epoch.
    """
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if loss_scaler is not None:
        checkpoint_path = output_dir / f'checkpoint-{epoch_name}.pth'
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args
        }

        if model_ema is not None:
            to_save['model_ema'] = get_state_dict(model_ema)
        
        save_on_master(to_save, checkpoint_path)
    else:
        client_state = {'epoch': epoch}

        if model_ema is not None:
            client_state['model_ema'] = get_state_dict(model_ema)

        model.save_checkpoint(
            save_dir=args.output_dir,
            tag=f'checkpoint-{epoch_name}',
            client_state=client_state
        )


def load_model(
    args,
    model_without_ddp,
    optimizer,
    loss_scaler,
    model_ema=None
):
    """
    Loads model from checkpoint if resume path is set.
    """
    if not args.resume:
        return
    
    checkpoint = (
        torch.hub.load_state_dict_from_url(
            args.resume, map_location='cpu', check_hash=True
        ) 
        if args.resume.startswith('https')
        else torch.load(args.resume, map_location='cpu')
    )

    model_without_ddp.load_state_dict(checkpoint['model'])
    print(f'Resumed checkpoint from {args.resume}')

    if 'optimizer' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

        if checkpoint['epoch'] != 'best':
            args.start_epoch = checkpoint['epoch'] + 1
        
        if (
            hasattr(args, 'model_ema') 
            and args.model_ema 
            and 'model_ema' in checkpoint
        ):
            _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])

        if 'scaler' in checkpoint:
            loss_scaler.load_state_dict(checkpoint['scaler'])
        
        print('Loaded optimizer, scaler, and EMA (if available)')


def all_reduce_mean(x):
    """
    Averages a scalar value across all distributed processes.
    """
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size

        return x_reduce.item()
    
    return x


class SmoothedValue(object):
    """
    Tracks a series of values and provides access to smoothed statistics
    over a fixed-size window and the global series.
    """
    def __init__(
        self,
        window_size=20,
        fmt='{median:.4f} ({global_avg:.4f})'
    ):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt


    def update(self, value, n=1):
        """
        Updates the tracker with a new value.
        """
        self.deque.append(value)
        self.count += n
        self.total += value * n


    def synchronize_between_processes(self):
        """
        Synchronizes total and count across distributed processes.
        """
        if not is_dist_avail_and_initialized():
            return
        
        t = torch.tensor([
            self.count, self.total
        ], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        self.count = int(t[0].item())
        self.total = t[1].item()


    @property
    def median(self):
        d = torch.tensor(list(self.deque))

        return d.median().item()
    

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)

        return d.mean().item()


    @property
    def global_avg(self):
        return self.total / self.count if self.count != 0 else 0.0


    @property
    def max(self):
        return max(self.deque)


    @property
    def value(self):
        return self.deque[-1]
    

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )
    

class MetricLogger(object):
    """
    Tracks and logs multiple training metrics.
    """
    def __init__(self, delimiter: str = '\t'):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter


    def update(self, **kwargs):
        """
        Updates metrics by name.
        """
        for k, v in kwargs.items():
            if v is None:
                continue

            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(v, (float, int)), \
                f"Value for metric '{k}' must be float or int, got {type(v)}"
            
            self.meters[k].update(v)


    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        
        return AttributeError(
            f"'{type(self).__name__}' object has no attribute '{attr}'"
        )
    

    def __str__(self):
        """
        Returns a string summary of all meters:
            metric_name: median (global_avg)
        """
        return self.delimiter.join(
            f'{name}: {str(meter)}'
            for name, meter in self.meters.items()
        )
    

    def get_mlm_acc(self):
        for name, meter in self.meters.items():
            if name == 'mlm_acc':
                print('avlue:', meter.value)
                print('avg:', meter.global_avg)

                return meter.global_avg
            

    def get_class_acc(self):
        for name, meter in self.meters.items():
            if name == 'class_acc':
                print('avlue:', meter.value)
                print('avg:', meter.global_avg)

                return meter.global_avg
    

    def add_meter(self, name, meter):
        """
        Adds a custom SmoothedValue meter manually.
        """
        self.meters[name] = meter


    def log_every(self, iterable, print_freq, header=None):
        """
        Wraps an iterable to log metrics and timing info every 
        `print_freq` steps.
        """
        MB = 1024.0 * 1024.0

        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()

        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')

        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]

        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj

            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i - 1)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(
                        log_msg.format(
                            i, len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i, len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time)
                        )
                    )

            i += 1
            end = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} '
              f'({total_time / len(iterable):.4f} s / it)')
        

    def synchronize_between_processes(self):
        """
        Synchronizes all meter statistics across distributed processes.
        """
        for meter in self.meters.values():
            meter.synchronize_between_processes()


class NativeScalerWithGradNormCount:
    state_dict_key = 'amp_scaler'


    def __init__(self):
        self._scaler = torch.amp.GradScaler()


    def __call__(
        self,
        loss,
        optimizer,
        clip_grad=None,
        parameters=None,
        create_graph=False,
        update_grad=True
    ):
        self._scaler.scale(loss).backward(create_graph=create_graph)

        grad_norm = None
        if update_grad:
            if clip_grad is not None and parameters is not None:
                self._scaler.unscale_(optimizer)
                grad_norm = nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                grad_norm = get_grad_norm_(parameters)

            self._scaler.step(optimizer)
            self._scaler.update()

        return grad_norm
    

    def state_dict(self):
        return self._scaler.state_dict()
    

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)
