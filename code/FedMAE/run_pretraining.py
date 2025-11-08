import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
import json
import time
import argparse
import datetime
from pathlib import Path
from copy import deepcopy

import torch
import numpy as np
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import utils.misc as misc
import FedMAE.models_mae as models_mae
from utils.start_config import print_options
from utils.weight_decay import add_weight_decay
from FedMAE.engine_for_pretraining import train_one_epoch
from utils.FedAvg_utils import Partial_Client_Selection, average_model
from utils.datasets import DatasetFLPretrain, create_dataset_and_evalmetrix
from utils.lora_utils import (
    fuse_adapters,
    add_lora_to_vit, 
    set_trainable_for_adapter_phase
)


def get_args():
    parser = argparse.ArgumentParser('FedMAE pre-training', add_help=False)

    parser.add_argument(
        '--batch_size', type=int, default=64,
        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus)'
    )
    parser.add_argument('--save_ckpt_freq', type=int, default=20)
    parser.add_argument(
        '--accum_iter', type=int, default=1,
        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)'
    )
    parser.add_argument(
        '--train_mode', type=str, default='pretrain',
        choices=['pretrain', 'finetune']
    )

    # Model parameters
    parser.add_argument('--model_name', type=str, default='mae')
    parser.add_argument(
        '--model', type=str, default='mae_vit_large_patch16',
        metavar='MODEL', help='Name of model to train'
    )
    parser.add_argument(
        '--input_size', type=int, default=224,
        help='Images input size'
    )
    parser.add_argument(
        '--mask_ratio', type=float, default=0.75,
        help='Masking ratio (percentage of removed patches)'
    )
    parser.add_argument(
        '--norm_pix_loss', action='store_true',
        help='Use (per-patch) normalized pixels as targets for computing loss'
    )
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument(
        '--weight_decay', type=float, default=0.05,
        help='Weight decay (default: 0.05)'
    )
    parser.add_argument(
        '--clip_grad', type=float, default=None,
        metavar='NORM',
        help='Clip gradient norm (default: None, no clipping)'
    )
    parser.add_argument(
        '--lr', type=float, default=None,
        metavar='LR',
        help='Learning rate (absolute lr)'
    )
    parser.add_argument(
        '--blr', type=float, default=1e-3,
        metavar='LR',
        help='Base learning rate: absolute_lr = base_lr * total_batch_size / 256'
    )
    parser.add_argument(
        '--min_lr', type=float, default=0.0,
        metavar='LR',
        help='Lower lr bound for cyclic schedulers that hit 0'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=40,
        metavar='N',
        help='Epochs to warmup LR'
    )

    # Dataset parameters
    parser.add_argument(
        '--dataset', type=str, default='Retina',
        help='Dataset for pre-training'
    )
    parser.add_argument(
        '--data_path', type=str, default='../../data/Retina',
        help='Dataset path'
    )
    parser.add_argument(
        '--output_dir', type=str, default='',
        help='Path where to save, empty for no saving'
    )
    parser.add_argument(
        '--log_dir', default=None,
        help='Path where to tensorboard log'
    )
    parser.add_argument(
        '--device', default='cuda',
        help='Device to use for training / testing'
    )
    parser.add_argument(
        '--seed', type=int, default=0
    )
    parser.add_argument(
        '--resume', default='',
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--start_epoch', type=int, default=0,
        metavar='N',
        help='Start epoch'
    )
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument(
        '--pin_mem', action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU'
    )
    parser.set_defaults(pin_mem=True)

    # Distributed training parameters
    parser.add_argument(
        '--world_size', type=int, default=1,
        help='Number of distributed processes'
    )
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--sync_bn', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument(
        '--dist_url', default='env://',
        help='url used to set up distributed training'
    )

    # FL related parameters
    parser.add_argument(
        '--n_clients', type=int, default=5,
        help='Number of total clients'
    )
    parser.add_argument(
        '--E_epoch', type=int, default=1,
        help='Local training epoch in FL'
    )
    parser.add_argument(
        '--max_communication_rounds', type=int, default=100,
        help='Total communication rounds'
    )
    parser.add_argument(
        '--num_local_clients', type=int, default=-1, choices=[10, -1],
        help='Number of selected clients per communication round, -1 means all clients'
    )
    parser.add_argument(
        '--split_type', type=str, default='central',
        help='Which data partition to use'
    )

    # LoRA related parameters
    parser.add_argument(
        '--lora_start_epoch', type=int, default=0,
        help='The communication round to switch from full-param warm-up to adapter only'
    )
    parser.add_argument('--lora_rank', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=float, default=0.0)
    parser.add_argument('--lora_weight_decay', type=float, default=0.01)
    parser.add_argument(
        '--lora_fuse_every', type=int, default=0,
        help='Fuse adapters into base every N epochs after lora_start_epoch'
    )

    return parser.parse_args()


def main(args, model):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    misc.fix_random_seeds(args)

    cudnn.benchmark = True

    # Prepare output_dir to save model checkpoints
    os.makedirs(args.output_dir, exist_ok=True)

    # Prepare dataset
    create_dataset_and_evalmetrix(args)

    # Configure the clients, prepare model, optimizer, loss_scaler for 
    # each client
    model_all, optimizer_all, loss_scaler_all = Partial_Client_Selection(
        args, model
    )
    model_avg = deepcopy(model).to(device)

    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print('=============== Running pre-training ===============')
    tot_clients = args.dis_csv_files
    print('total_clients:', tot_clients)
    epoch = args.start_epoch - 1

    print(
        f'Start training for {args.max_communication_rounds} epochs, '
        f'distributed={args.distributed}'
    )
    start_time = time.time()

    def unwrap_model(model):
        return model.module if hasattr(model, 'module') else model

    while True:
        epoch += 1
        print('epoch: ', epoch)
        adapter_phase = epoch >= args.lora_start_epoch
        if adapter_phase:
            print('=== Adapter-only training phase ===')
        else:
            print('=== Warm-up full-parameter training phase ===')

        if epoch == args.lora_start_epoch:
            gm = unwrap_model(model_avg)
            add_lora_to_vit(
                model=gm,
                r=args.lora_rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout
            )
            set_trainable_for_adapter_phase(gm)
            # Share identical LoRA initialization across all clients
            global_state = gm.state_dict()

            for cid in args.proxy_clients:
                cm = unwrap_model(model_all[cid])
                add_lora_to_vit(
                    model=cm,
                    r=args.lora_rank,
                    alpha=args.lora_alpha,
                    dropout=args.lora_dropout
                )
                cm.load_state_dict(global_state, strict=False)

                # Freeze base, enable adapters/LN/bias only
                set_trainable_for_adapter_phase(cm)

                if args.distributed:
                    model_all[cid] = torch.nn.parallel.DistributedDataParallel(
                        cm,
                        device_ids=[args.gpu],
                        find_unused_parameters=False
                    )
                    optim_target = model_all[cid].module
                else:
                    model_all[cid] = cm
                    optim_target = cm

                # Rebuild optimizer over trainable parameters only
                param_groups = add_weight_decay(
                    optim_target, 
                    weight_decay=args.weight_decay,
                    lora_weight_decay=args.lora_weight_decay
                )

                optimizer_all[cid] = torch.optim.AdamW(
                    param_groups,
                    lr=args.lr,
                    betas=(0.9, 0.95)
                )
        
        # Randomly select partial clients
        if args.num_local_clients == len(args.dis_csv_files): # all clients
            cur_selected_clients = args.proxy_clients
        else: # partial clients
            cur_selected_clients = np.random.choice(
                tot_clients, args.num_local_clients, replace=False
            ).tolist()
        
        # Get the quantity of clients joined in the FL train for 
        # updating the clients weights
        cur_tot_client_lens = 0
        for client in cur_selected_clients:
            cur_tot_client_lens += args.clients_with_len[client]
        
        for cur_single_client, proxy_single_client in zip(
            cur_selected_clients, args.proxy_clients
        ):
            print('cur_single_client: ', cur_single_client)
            print('proxy_single_client: ', proxy_single_client)
            
            args.single_client = cur_single_client
            args.client_weights[proxy_single_client] = (
                args.clients_with_len[cur_single_client] / cur_tot_client_lens
            )
            
            # Get dataset for each client for pretraining
            dataset_train = DatasetFLPretrain(args)
            
            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()
            sampler_rank = global_rank
            num_training_steps_per_inner_epoch = (
                len(dataset_train) // args.batch_size // num_tasks
            )

            print(f'========= client: {proxy_single_client} =========')
            if args.distributed:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, 
                    rank=sampler_rank, shuffle=True
                )
            else:    
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
            
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, 
                sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True
            )
            
            # Prepare model for a client
            model = model_all[proxy_single_client]
            optimizer = optimizer_all[proxy_single_client]
            loss_scaler = loss_scaler_all[proxy_single_client]
            
            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch)

            n_parameters = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )  
            total_batch_size = (
                args.batch_size * args.accum_iter * misc.get_world_size()
            )
            
            if args.lr is None: # only base_lr is specified
                args.lr = args.blr * total_batch_size / 256

            print('base lr: %.2e' % (args.lr * 256 / total_batch_size))
            print('actual lr: %.2e' % args.lr)
            print('accumulate grad iterations: %d' % args.accum_iter)
            print('effective batch size: %d' % total_batch_size)
            print(
                'Number of training steps = %d' % num_training_steps_per_inner_epoch
            )
            print(
                'Number of training examples per epoch = %d' % (total_batch_size * num_training_steps_per_inner_epoch)
            )
            
            for inner_epoch in range(args.E_epoch):
                # ========= training one epoch of MAE =========
                train_stats = train_one_epoch(
                    model, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    proxy_single_client=proxy_single_client,
                    log_writer=log_writer,
                    args=args
                )
                
                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    'client': args.single_client,
                    'epoch': epoch,
                    'inner_epoch': inner_epoch,
                    'n_parameters': n_parameters
                }

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    
                    with open(
                        os.path.join(
                            args.output_dir, 'log.txt'
                        ), mode='a', encoding='utf-8'
                    ) as f:
                        f.write(json.dumps(log_stats) + '\n')
            
        # Average model
        average_model(args, model_avg, model_all)

        # Fuse adapters
        if (
            epoch > args.lora_start_epoch 
            and args.lora_fuse_every > 0
        ):
            if (epoch - args.lora_start_epoch + 1) % args.lora_fuse_every == 0:
                gm = unwrap_model(model_avg)
                print(f'Fusing adapters into base at epoch {epoch}')
                fuse_adapters(gm)

                gsd = gm.state_dict()
                for cid in args.proxy_clients:
                    cm = unwrap_model(model_all[cid])
                    cm.load_state_dict(gsd)

        # Save the global model
        if args.output_dir and (epoch + 1) % args.save_ckpt_freq == 0:
            misc.save_model(
                args=args, model=model_avg, model_without_ddp=model_avg,
                optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
            )

        # End criterion
        if (
            args.global_step_per_client[proxy_single_client] 
            >= args.t_total[proxy_single_client]
        ):
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()

    print('=============== End of pre-training ===============')
    print('Pretraining time: {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()

    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = models_mae.__dict__[opts.model](norm_pix_loss=opts.norm_pix_loss)
    print_options(opts, model)
    
    # Set train val related paramteres
    opts.best_mlm_acc = {}
    opts.current_mlm_acc = {}
    
    # Run pretraining
    main(opts, model)
