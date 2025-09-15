"""
This module provides data augmentation techniques for pretraining and
finetuning in self-supervised federated learning. 

Originally adapted from the SSL-FL codebase:
https://github.com/rui-yan/SSL-FL
"""

from PIL import Image
Image.LOAD_TRUNCATED_IMAGES = True

import torch
from torchvision import transforms


RETINA_MEAN = (0.5007, 0.5010, 0.5019)
RETINA_STD = (0.0342, 0.0535, 0.0484)


class DataAugmentationForPretrain(object):
    """
    Data augmentations for pretraining.
    """
    def __init__(self, args):
        if args.dataset == 'Retina':
            mean, std = RETINA_MEAN, RETINA_STD
        else:
            mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

        if args.model_name == 'mae':
            if args.dataset == 'Retina':
                self.common_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        args.input_size, scale=(0.2, 1.0), interpolation=3
                    ), 
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.4, 0.4, 0.4),
                    transforms.RandomHorizontalFlip(p=0.5)
                ])
            elif args.dataset == 'COVID-FL':
                self.common_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        args.input_size, scale=(0.4, 1.0), interpolation=3
                    ),
                    transforms.ColorJitter(hue=0.05, saturation=0.05),
                    transforms.RandomHorizontalFlip(p=0.5)
                ])
            else:
                self.common_transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        args.input_size, scale=(0.2, 1.0), interpolation=3
                    ), 
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ColorJitter(0.1, 0.1, 0.1),
                    transforms.RandomHorizontalFlip(p=0.5)
                ])

        self.patch_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor(mean),
                std=torch.tensor(std)
            )
        ])

        self.args = args

    
    def __call__(self, image):
        if self.args.model_name == 'mae':
            for_patches = self.common_transform(image)

            return self.patch_transform(for_patches)
        

    def __repr__(self):
        if self.args.model_name == 'mae':
            repr = 'DataAugmentationForMae,\n'
            repr += f'  common_transform = {str(self.common_transform)},\n'
            repr += f'  patch_transform = {str(self.patch_transform)}\n'

        return repr
    

def build_transform(is_train, mode, args):
    """
    Builds data augmentation transforms for finetuning.
    """
    if args.dataset == 'Retina':
        mean, std = RETINA_MEAN, RETINA_STD
    else:
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    if mode == 'finetune':
        if is_train:
            if args.dataset == 'COVID-FL':
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        args.input_size, scale=(0.8, 1.2)
                    ),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std)
                    )
                ])
            else:
                transform = transforms.Compose([
                    transforms.RandomResizedCrop(
                        args.input_size, scale=(0.6, 1.0)
                    ),
                    transforms.RandomRotation(degrees=10),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=torch.tensor(mean),
                        std=torch.tensor(std)
                    )
                ])
        else:
            transform = transforms.Compose([
                transforms.Resize(size=args.input_size),
                transforms.CenterCrop(size=(args.input_size, args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std)
                )
            ])

    return transform
