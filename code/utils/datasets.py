"""
This module defines dataset classes and utility functions for
self-supervised federated learning.

Originally adapted from the SSL-FL codebase:
https://github.com/rui-yan/SSL-FL
"""

import os
from PIL import Image

import numpy as np
import pandas as pd
import torch.utils.data as data
from skimage.transform import resize

from .data_augmentation import DataAugmentationForPretrain, build_transform


class DatasetFLPretrain(data.Dataset):
    """
    Dataset for self-supervised federated pretraining.
    """
    def __init__(self, args):
        super(DatasetFLPretrain, self).__init__()

        if args.split_type == 'central':
            cur_client_path = os.path.join(
                args.data_path, args.split_type, args.single_client
            )
        else:
            cur_client_path = os.path.join(
                args.data_path, 
                f'{args.n_clients}_clients',
                args.split_type,
                args.single_client
            )

        self.img_names = list({
            line.strip().split(',')[0] for line in open(cur_client_path)
        })

        self.labels = {
            line.strip().split(',')[0]: float(line.strip().split(',')[1])
            for line in open(os.path.join(args.data_path, 'labels.csv'))
        }

        self.transform = DataAugmentationForPretrain(args)
        self.args = args

    
    def __len__(self):
        return len(self.img_names)
    

    def __getitem__(self, index):
        index = index % len(self.img_names)
        img_path = os.path.join(
            self.args.data_path, 'train', self.img_names[index]
        )

        name = self.img_names[index]
        target = self.labels[name]
        target = np.asarray(target).astype('int64')

        if self.args.dataset == 'Retina':
            img = np.load(img_path)
            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(img_path).convert('RGB'))

        if img.ndim < 3:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        if self.transform is not None:
            img = Image.fromarray(np.uint8(img))
            sample = self.transform(img)

        return sample, target
    

class DatasetFLFinetune(data.Dataset):
    """
    Dataset for self-supervised federated fine-tuning.
    """
    def __init__(self, args, phase, mode='finetune'):
        super(DatasetFLFinetune, self).__init__()

        self.phase = phase
        is_train = (phase == 'train')

        if not is_train:
            args.single_client = os.path.join(
                args.data_path, f'{self.phase}.csv'
            )

        if args.split_type == 'central':
            cur_client_path = os.path.join(
                args.data_path, args.split_type, args.single_client
            )
        else:
            cur_client_path = os.path.join(
                args.data_path, 
                f'{args.n_clients}_clients',
                args.split_type,
                args.single_client
            )

        self.img_paths = list({
            line.strip().split(',')[0] for line in open(cur_client_path)
        })
        self.labels = {
            line.strip().split(',')[0]: float(line.strip().split(',')[1])
            for line in open(os.path.join(args.data_path, 'labels.csv'))
        }

        self.transform = build_transform(is_train, mode, args)
        self.args = args


    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, index):
        index = index % len(self.img_paths)
        img_path = os.path.join(
            self.args.data_path, self.phase, self.img_paths[index]
        )

        name = self.img_paths[index]
        target = self.labels[name]
        target = np.asarray(target).astype('int64')

        if self.args.dataset == 'Retina':
            img = np.load(img_path)
            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(img_path).convert('RGB'))

        if img.ndim < 3:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)

        return sample, target
    

def create_dataset_and_evalmetrix(args, mode='pretrain'):
    """
    Creates dataset and evaluation metrics for each client.
    """
    if args.split_type == 'central':
        args.dis_csv_files = os.listdir(
            os.path.join(args.data_path, args.split_type)
        )
    else:
        args.dis_csv_files = os.listdir(
            os.path.join(
                args.data_path, f'{args.n_clients}_clients', args.split_type
            )
        )

    args.clients_with_len = {}

    for single_client in args.dis_csv_files:
        if args.split_type == 'central':
            img_paths = list({
                line.strip().split(',')[0] 
                for line in open(
                    os.path.join(
                        args.data_path, args.split_type, single_client
                    )
                )
            })
        else:
            img_paths = list({
                line.strip().split(',')[0] 
                for line in open(
                    os.path.join(
                        args.data_path, 
                        f'{args.n_clients}_clients', 
                        args.split_type, 
                        single_client
                    )
                )
            })

        args.clients_with_len[single_client] = len(img_paths)

    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_csv_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_csv_files)
    args.save_model = False
    args.best_eval_loss = {}

    for single_client in args.dis_csv_files:
        if mode == 'pretrain':
            args.best_mlm_acc[single_client] = 0
            args.current_mlm_acc[single_client] = []
        
        if mode == 'finetune':
            args.best_acc[single_client] = 0 if args.nb_classes > 1 else 999
            args.current_acc[single_client] = 0
            args.current_test_acc[single_client] = []
            args.best_eval_loss[single_client] = 9999
