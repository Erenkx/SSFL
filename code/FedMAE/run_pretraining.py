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
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import utils.misc as misc
import FedMAE.models_mae as models_mae
from utils.start_config import print_options
from FedMAE.engine_for_pretraining import train_one_epoch
from utils.FedAvg_utils import Partial_Client_Selection, average_model
from utils.datasets import DatasetFLPretrain, create_dataset_and_evalmetrix


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
    parser.add_argument('--local_rank', type=int, default=1)
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
    parser.add_argument(
        '--topk_ratio', type=float, default=1.0,
        help='Top-k sparsification ratio'
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

    while True:
        epoch += 1
        print('epoch: ', epoch)
        
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
