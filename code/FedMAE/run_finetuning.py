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
import FedMAE.models_vit as models_vit
from utils.start_config import print_options
from FedMAE.engine_for_finetuning import train_one_epoch
from utils.FedAvg_utils import Partial_Client_Selection, valid, average_model
from utils.datasets import DatasetFLFinetune, create_dataset_and_evalmetrix


def get_args():
    parser = argparse.ArgumentParser(
        'FedMAE fine-tuning for image classification', add_help=False
    )

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
        '--model', type=str, default='vit_large_patch16',
        metavar='MODEL', 
        help='Name of model to train'
    )
    parser.add_argument(
        '--input_size', type=int, default=224,
        help='Images input size'
    )
    parser.add_argument(
        '--drop_path', type=float, default=0.1,
        metavar='PCT',
        help='Drop path rate (default: 0.1)'
    )
    parser.add_argument(
        '--disable_eval_during_finetuning', default=False, action='store_true'
    )

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
        '--min_lr', type=float, default=1e-6,
        metavar='LR',
        help='Lower lr bound for cyclic schedulers that hit 0'
    )
    parser.add_argument(
        '--warmup_epochs', type=int, default=5,
        metavar='N',
        help='Epochs to warmup LR'
    )
    parser.add_argument(
        '--layer_decay', type=float, default=0.75,
        help='Layer-wise lr decay from ELECTRA/BEiT'
    )

    # Augmentation parameters
    parser.add_argument(
        '--color_jitter', type=float, default=None, 
        metavar='PCT',
        help='Color jitter factor (enabled only when not using Auto/RandAug)'
    )
    parser.add_argument(
        '--aa', type=str, default='rand-m9-mstd0.5-inc1', 
        metavar='NAME',
        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'
    )
    parser.add_argument(
        '--smoothing', type=float, default=0.1,
        help='Label smoothing (default: 0.1)'
    )

    # * Random Erase params
    parser.add_argument(
        '--reprob', type=float, default=0.25, 
        metavar='PCT',
        help='Random erase prob (default: 0.25)'
    )
    parser.add_argument(
        '--remode', type=str, default='pixel',
        help='Random erase mode (default: "pixel")'
    )
    parser.add_argument(
        '--recount', type=int, default=1,
        help='Random erase count (default: 1)'
    )
    parser.add_argument(
        '--resplit', action='store_true', default=False,
        help='Do not random erase first (clean) augmentation split'
    )

    # * Mixup params
    parser.add_argument(
        '--mixup', type=float, default=0,
        help='Mixup alpha, mixup enabled if > 0'
    )
    parser.add_argument(
        '--cutmix', type=float, default=0,
        help='Cutmix alpha, cutmix enabled if > 0'
    )
    parser.add_argument(
        '--cutmix_minmax', type=float, nargs='+', default=None,
        help='Cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)'
    )
    parser.add_argument(
        '--mixup_prob', type=float, default=1.0,
        help='Probability of performing mixup or cutmix when either/both is enabled'
    )
    parser.add_argument(
        '--mixup_switch_prob', type=float, default=0.5,
        help='Probability of switching to cutmix when both mixup and cutmix enabled'
    )
    parser.add_argument(
        '--mixup_mode', type=str, default='batch',
        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"'
    )

    # * Finetuning params
    parser.add_argument(
        '--finetune', default='',
        help='Finetune from checkpoint'
    )
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool='avg')
    parser.add_argument(
        '--cls_token', action='store_false', dest='global_pool',
        help='Use class token instead of global pool for classification'
    )

    # Dataset parameters
    parser.add_argument(
        '--dataset', type=str, default='Retina',
        choices=['Retina', 'Derm', 'COVID-FL'],
        help='Dataset for fine-tuning'
    )
    parser.add_argument(
        '--data_path', type=str, default='../../data/Retina',
        help='Dataset path'
    )
    parser.add_argument(
        '--nb_classes', type=int, default=2,
        help='Number of the classification types'
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
    parser.add_argument(
        '--eval', action='store_true',
        help='Perform evaluation only'
    )
    parser.add_argument(
        '--dist_eval', default=False, action='store_true',
        help='Enabling distributed evaluation (recommended during training for faster monitor)'
    )
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument(
        '--pin_mem', action='store_true',
        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU'
    )
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    
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

    return parser.parse_args()


def main(args, model):
    misc.init_distributed_mode(args)

    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print('{}'.format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # Fix the seed for reproducibility
    misc.fix_random_seeds(args)

    cudnn.benchmark = True

    # Prepare dataset
    create_dataset_and_evalmetrix(args, mode='finetune')

    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val = DatasetFLFinetune(args=args, phase='test')

    if args.eval:
        dataset_test = DatasetFLFinetune(args=args, phase='test')
    else:
        dataset_test = None

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print(
                'Warning: Enabling distributed evaluation with an eval '
                'dataset not divisible by process number.\n'
                'This will slightly alter validation results as extra '
                'duplicate entries are added to achieve equal num of '
                'samples per-process.'
            )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, 
                num_replicas=num_tasks, rank=global_rank, shuffle=True
            ) # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
    else:
        data_loader_test = None

    # Configure the clients, prepare the model, criterion, optimizer for
    # each client
    model_all, optimizer_all, criterion_all, loss_scaler_all, mixup_fn_all = (
        Partial_Client_Selection(args, model, mode='finetune')
    )
    model_avg = deepcopy(model).to(device)

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    print('=============== Running fine-tuning ===============')
    tot_clients = args.dis_csv_files
    print('total_clients: ', tot_clients)
    epoch = args.start_epoch - 1

    start_time = time.time()
    max_accuracy = 0.0

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

            # Get dataset for each client for pretraining finetuning 
            dataset_train = DatasetFLFinetune(args=args, phase='train')

            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()

            print(f'========= client: {proxy_single_client} =========')
            if args.distributed:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, 
                    shuffle=True
                )
            else:    
                sampler_train = torch.utils.data.RandomSampler(dataset_train)

            print('Sampler_train = %s' % str(sampler_train))

            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
            )

            # Prepare model for a client
            model = model_all[proxy_single_client]
            optimizer = optimizer_all[proxy_single_client]
            criterion = criterion_all[proxy_single_client]
            loss_scaler = loss_scaler_all[proxy_single_client]
            mixup_fn = mixup_fn_all[proxy_single_client]

            if args.distributed:
                model_without_ddp = model.module
            else:
                model_without_ddp = model

            n_parameters = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            total_batch_size = (
                args.batch_size * args.accum_iter * misc.get_world_size()
            )
            num_training_steps_per_inner_epoch = (
                len(dataset_train) // total_batch_size
            )
            print('LR = %.8f' % args.lr)
            print('Batch size = %d' % total_batch_size)
            print('Number of training examples = %d' % len(dataset_train))
            print(
                'Number of training training per epoch = %d' %
                num_training_steps_per_inner_epoch
            )

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch)

            if args.eval:
                misc.load_model(
                    args=args, model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler, 
                    model_ema=None
                )

                test_stats = valid(args, model, data_loader_test)
                print(
                    f'Accuracy of the network on the {len(dataset_test)} '
                    f"test images: {test_stats['acc1']:.1f}%"
                )
                model.cpu()
                exit(0)

            for inner_epoch in range(args.E_epoch):
                # ========= training one epoch of MAE =========
                train_stats = train_one_epoch(
                    model, criterion, data_loader_train,
                    optimizer, device, epoch, loss_scaler,
                    args.clip_grad, proxy_single_client,
                    mixup_fn,
                    log_writer=log_writer,
                    args=args
                )

                log_stats = {
                    **{f'train_{k}': v for k, v in train_stats.items()},
                    'client': cur_single_client,
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
        if args.output_dir:
            if (
                (epoch + 1) % args.save_ckpt_freq == 0
                or (epoch + 1) == args.max_communication_rounds
            ):
                misc.save_model(
                    args=args, model=model_avg, model_without_ddp=model_avg,
                    optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch
                )
        
        if data_loader_val is not None:
            model_avg.to(args.device)
            test_stats = valid(args, model_avg, data_loader_val)
            print(
                f'Accuracy of the network on the {len(dataset_val)} '
                f"validation images: {test_stats['acc1']:.1f}%"
            )

            if max_accuracy < test_stats['acc1']:
                max_accuracy = test_stats['acc1']
                if args.output_dir:
                    misc.save_model(
                        args=args, model=model_avg, 
                        model_without_ddp=model_without_ddp, 
                        optimizer=optimizer,
                        loss_scaler=loss_scaler, epoch='best', model_ema=None
                    )
                
            print(f'Max accuracy: {max_accuracy:.2f}%')
            if log_writer is not None:
                log_writer.update(
                    test_acc1=test_stats['acc1'], head="perf", step=epoch
                )
                log_writer.update(
                    test_acc5=test_stats['acc5'], head="perf", step=epoch
                )
                log_writer.update(
                    test_loss=test_stats['loss'], head="perf", step=epoch
                )

            log_stats = {
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
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
                f.write(json.dumps(log_stats) + "\n")
        
        model_avg.to('cpu')
        
        print(
            'global_step_per_client:', 
            args.global_step_per_client[proxy_single_client]
        )
        print('t_total:', args.t_total[proxy_single_client])

        if (
            args.global_step_per_client[proxy_single_client] 
            >= args.t_total[proxy_single_client]
        ):
            break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    if args.distributed and dist.is_initialized():
        dist.destroy_process_group()

    print('=============== End of fine-tuning ===============')
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize model
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
    )
    print_options(args, model)
    
    # Set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    # Run finetuning
    main(args, model)
