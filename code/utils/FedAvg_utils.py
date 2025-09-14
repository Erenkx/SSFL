from __future__ import absolute_import, division, print_function

import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from timm.utils import accuracy
from timm.data.mixup import Mixup
from timm.models.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from . import misc as misc
from .lars import LARS
from .lr_decay import param_groups_lrd
from .pos_embed import interpolate_pos_embed
from .misc import NativeScalerWithGradNormCount as NativeScaler
from .optim_factory import create_optimizer, add_weight_decay, \
    LayerDecayValueAssigner


def Partial_Client_Selection(args, model, mode='pretrain'):
    device = torch.device(args.device)

    if args.num_local_clients == -1:
        args.proxy_clients = args.dis_csv_files
        args.num_local_clients = len(args.dis_csv_files)
    else:
        args.proxy_clients = [
            'train_' + str(i) for i in range(args.num_local_clients)
        ]

    model_all = {}
    optimizer_all = {}
    criterion_all = {}
    lr_scheduler_all = {}
    wd_scheduler_all = {}
    loss_scaler_all = {}
    mixup_fn_all = {}
    args.learning_rate_record = {}
    args.t_total = {}

    if (mode == 'finetune' or mode == 'linprob') and args.finetune:
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True
            )
        else:
            checkpoint = torch.load(
                args.finetune, map_location='cpu', weights_only=False
            )
        print(f'Loaded pretrained checkpoint from {args.finetune}')

        if args.model_name == 'mae':
            checkpoint_model = checkpoint['model']
            decoder_related_keys = [
                'decoder_', 'mask_token', 'decoder_pos_embed',
                'decoder_pred', 'decoder_blocks', 'decoder_embed',
                'decoder_norm'
            ]
            checkpoint_model = {
                k: v
                for k, v in checkpoint_model.items()
                if not any(
                    k.startswith(prefix) for prefix in decoder_related_keys
                )
            }

        state_dict = model.state_dict()
        for k in ['head_weight', 'head_bias']:
            if k in checkpoint_model \
                and checkpoint_model[k].shape == state_dict[k].shape:
                print(f'Removing key {k} from pretrained checkpoint')
                del checkpoint_model[k]

        interpolate_pos_embed(model, checkpoint_model)

        if args.model_name == 'mae':
            msg = model.load_state_dict(checkpoint_model, strict=False)
            print(msg)

            if args.global_pool:
                assert set(msg.missing_keys) == {
                    'head.weight', 'head.bias',
                    'fc_norm.weight', 'fc_norm.bias'
                }
            else:
                assert set(msg.missing_keys) == {
                    'head.weight', 'head.bias'
                }

        if mode == 'finetune':
            trunc_normal_(model.head.weight, std=2e-5)
        elif mode == 'linprob':
            trunc_normal_(model.head.weight, std=0.01)

            model.head = torch.nn.Sequential(
                torch.nn.BatchNorm1d(
                    model.head.in_features,
                    affine=False,
                    eps=1e-6
                ),
                model.head
            )

            for _, p in model.named_parameters():
                p.requires_grad = False
            for _, p in model.head.named_parameters():
                p.requires_grad = True

    if args.distributed:
        if args.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    for proxy_single_client in args.proxy_clients:
        num_tasks = misc.get_world_size()

        print('clients_with_len:', args.clients_with_len[proxy_single_client])

        if args.model_name == 'mae':
            total_batch_size = args.batch_size * args.accum_iter * num_tasks
            if args.lr is None:
                args.lr = args.blr * total_batch_size / 256

        num_training_steps_per_inner_epoch = (
            args.clients_with_len[proxy_single_client] // total_batch_size
        )

        print(f'Batch size = {total_batch_size}')
        print('Number of training steps = {}'.format(
            num_training_steps_per_inner_epoch
        ))
        print('Number of training examples per epoch = {}'.format(
            total_batch_size * num_training_steps_per_inner_epoch
        ))

        model_all[proxy_single_client] = deepcopy(model)
        model_all[proxy_single_client] = model_all[proxy_single_client].to(
            device
        )

        if args.distributed:
            model_all[proxy_single_client] = (
                nn.parallel.DistributedDataParallel(
                    model_all[proxy_single_client],
                    device_ids=[args.gpu],
                    find_unused_parameters=True
                )
            )

        if args.distributed:
            model_without_ddp = model_all[proxy_single_client].module
        else:
            model_without_ddp = model_all[proxy_single_client]

        if mode =='pretrain':
            if args.model_name == 'mae':
                param_groups = add_weight_decay(
                    model_without_ddp, args.weight_decay
                )
                optimizer_all[proxy_single_client] = torch.optim.AdamW(
                    param_groups,
                    lr=args.lr,
                    betas=(0.9, 0.95)
                )
        elif mode == 'finetune':
            if args.model_name == 'mae':
                param_groups = param_groups_lrd(
                    model_without_ddp,
                    args.weight_decay,
                    no_weight_decay_list=model_without_ddp.no_weight_decay(),
                    layer_decay=args.layer_decay
                )
                optimizer_all[proxy_single_client] = torch.optim.AdamW(
                    param_groups,
                    lr=args.lr
                )
        elif mode == 'linprob':
            if args.model_name == 'mae':
                optimizer_all[proxy_single_client] = LARS(
                    model_without_ddp.head.parameters(),
                    lr=args.lr,
                    weight_decay=args.weight_decay
                )

        if mode == 'finetune':
            mixup_fn = None
            mixup_active = (
                args.mixup > 0 
                or args.cutmix > 0.0
                or args.cutmix_minmax is not None
            )
            if mixup_active:
                print('Mixup is activated!')
                mixup_fn = Mixup(
                    mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                    cutmix_minmax=args.cutmix_minmax,
                    prob=args.mixup_prob, switch_prob=args.mixup_switch_prob,
                    mode=args.mixup_mode,
                    label_smoothing=args.smoothing,
                    num_classes=args.nb_classes
                )
            mixup_fn_all[proxy_single_client] = mixup_fn

            if mixup_fn is not None:
                criterion = SoftTargetCrossEntropy()
            elif args.smoothing > 0.0:
                criterion = LabelSmoothingCrossEntropy(
                    smoothing=args.smoothing
                )
            else:
                criterion = nn.CrossEntropyLoss()
            criterion_all[proxy_single_client] = criterion

        if mode == 'linprob':
            criterion_all[proxy_single_client] = nn.CrossEntropyLoss()

        loss_scaler_all[proxy_single_client] = NativeScaler()

        args.t_total[proxy_single_client] = (
            num_training_steps_per_inner_epoch
            * args.E_epoch * args.max_communication_rounds
        )

        args.learning_rate_record[proxy_single_client] = []

    args.clients_weightes = {}
    args.global_step_per_client = {name: 0 for name in args.proxy_clients}

    if args.model_name == 'mae':
        if mode == 'pretrain':
            return model_all, optimizer_all, loss_scaler_all
        else:
            return (
                model_all, optimizer_all, criterion_all,
                loss_scaler_all, mixup_fn_all
            )
        

def average_model(args, model_avg, model_all):
    model_avg.cpu()
    print('Calculating the model average...')
    params = dict(model_avg.named_parameters())

    for name, param in params.items():
        for client in range(len(args.proxy_clients)):
            single_client = args.proxy_clients[client]

            single_client_weight = args.clients_weightes[single_client]
            single_client_weight = torch.from_numpy(
                np.array(single_client_weight)
            ).float()

            if client == 0:
                if args.distributed:
                    tmp_param_data = dict(
                        model_all[single_client].module.named_parameters()
                    )[name].data * single_client_weight
                else:
                    tmp_param_data = dict(
                        model_all[single_client].named_parameters()
                    )[name].data * single_client_weight
            else:
                if args.distributed:
                    tmp_param_data += dict(
                        model_all[single_client].module.named_parameters()
                    )[name].data * single_client_weight
                else:
                    tmp_param_data += dict(
                        model_all[single_client].named_parameters()
                    )[name].data * single_client_weight

        params[name].data.copy_(tmp_param_data)

    print('Updating each client model parameters...')

    for single_client in args.proxy_clients:
        if args.distributed:
            tmp_params = dict(
                model_all[single_client].module.named_parameters()
            )
        else:
            tmp_params = dict(
                model_all[single_client].named_parameters()
            )

        for name, param in params.items():
            tmp_params[name].data.copy_(param.data)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def  simple_accuracy(preds, labels):
    return (preds == labels).mean()


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    client_name = os.path.basename(args.single_client).split('.')[0]
    model_checkpoint = os.path.join(
        args.output_dir, f'{args.name}_{client_name}_checkpoint.bin'
    )
    torch.save(model_to_save.state_dict(), model_checkpoint)
    print(f'Saved model checkpoint to {model_checkpoint}')


def valid(args, model, data_loader):
    criterion = nn.CrossEntropyLoss()
    metric_logger = misc.MetricLogger(delimiter='  ')
    header = 'Test:'

    model.eval()
    print('++++++ Running validation ++++++')
    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        with torch.no_grad():
            output = model(images)
            loss = criterion(output, target)

        acc1, _ = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
        top1=metric_logger.acc1, losses=metric_logger.loss
    ))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def metric_evaluation(args, eval_result):
    if args.nb_classes == 1:
        if args.best_acc[args.single_client] < eval_result:
            Flag = False
        else:
            Flag = True
    else:
        if args.best_acc[args.single_client] < eval_result:
            Flag = True
        else:
            Flag = False

    return Flag
