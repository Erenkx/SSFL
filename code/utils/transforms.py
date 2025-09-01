"""
Custom image transformation utilities used in federated self-supervised
learning.

It includes tensor and numpy converters, and a data augmentation 
transform that performs random resized cropping with two output sizes.

Originally adapted from the SSL-FL codebase:
https://github.com/rui-yan/SSL-FL
"""

import math
import random
import warnings

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image


class ToNumpy:
    """
    Converts a PIL image to a NumPy array in CHW format.
    """
    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)

        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2) # HWC to CHW

        return np_img
    

class ToTensor:
    """
    Converts a PIL image to a PyTorch tensor in CHW format.

    Args:
        dtype (torch.dtype): Desired tensor dtype. Default is 
            torch.float32.
    """
    def __init__(self, dtype=torch.float32):
        self.dtype = dtype


    def __call__(self, pil_img):
        np_img = np.array(pil_img, dtype=np.uint8)

        if np_img.ndim < 3:
            np_img = np.expand_dims(np_img, axis=-1)
        np_img = np.rollaxis(np_img, 2) # HWC to CHW

        return torch.from_numpy(np_img).to(dtype=self.dtype)
    

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
    Image.HAMMING: 'PIL.Image.HAMMING',
    Image.BOX: 'PIL.Image.BOX'
}

_RANDOM_INTERPOLATION = (Image.BILINEAR, Image.BICUBIC)


def _pil_interp(method: str) -> int:
    """
    Converts a string interpolation method name to PIL's enum value.

    Args:
        method (str): Interpolation type.
            Options: 'bicubic', 'lanczos', 'hamming', or others.

    Returns:
        int: PIL.Image interpolation enum.
    """
    if method == 'bicubic':
        return Image.BICUBIC
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        return Image.BILINEAR
    

class RandomResizedCropAndInterpolationWithTwoPic:
    """
    Random resized crop with optional dual-size output and interpolation
    choices.

    Args:
        size (int or tuple): Target size of the first output.
        second_size (int or tuple): Second output size.
        scale (tuple): Range of crop area as a fraction of original 
            image area.
        ratio (tuple): Range of aspect ratios for crop.
        interpolation (str): Interpolation type for first crop. Default
            is 'bilinear'.
        second_interpolation (str): Interpolatioin type for second crop.
            Default is 'lanczos'.
    """
    def __init__(
        self,
        size,
        second_size=None,
        scale=(0.08, 1.0),
        ratio=(3 / 4, 4 / 3),
        interpolation='bilinear',
        second_interpolation='lanczos'
    ):
        self.size = size if isinstance(size, tuple) else (size, size)
        self.second_size = (
            second_size if isinstance(second_size, tuple) 
            else (second_size, second_size)
        ) if second_size is not None else None

        if scale[0] > scale[1] or ratio[0] > ratio[1]:
            warnings.warn('Scale or ratio range should be (min, max).')
        
        self.scale = scale
        self.ratio = ratio

        if interpolation == 'random':
            self.interpolation = _RANDOM_INTERPOLATION
        else:
            self.interpolation = _pil_interp(interpolation)
        
        self.second_interpolation = _pil_interp(second_interpolation)


    @staticmethod
    def get_params(img, scale, ratio):
        """
        Calculates random crop coordinates for a resized crop.

        Args:
            img (PIL.Image): Input image.
            scale (tuple): Min and max area of the crop as a fraction of
                image area.
            ratio (tuple): Min and max aspect ratio.

        Returns:
            tuple[int, int, int, int]: i, j, h, w crop coordinates.
        """
        width = img.size[0]
        height = img.size[1]
        area = width * height

        for _ in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= width and h <= height:
                i = random.randint(0, height - h)
                j = random.randint(0, width - w)

                return i, j, h, w
            
        original_ratio = width / height
        if original_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif original_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height

        i = (height - h) // 2
        j = (width - w) // 2

        return i, j, h, w
    

    def __call__(self, img):
        """
        Applies the random resized crop to the input image.

        Args:
            img (PIL.Image): Input image.

        Returns:
            PIL.Image or tuple[PIL.Image, PIL.Image]: Single or dual 
                resized crops.
        """
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        interpolation = (
            random.choice(self.interpolation)
            if isinstance(self.interpolation, (tuple, list))
            else self.interpolation
        )

        if self.second_size is None:
            return F.resized_crop(img, i, j, h, w, self.size, interpolation)
        else:
            return (
                F.resized_crop(img, i, j, h, w, self.size, interpolation),
                F.resized_crop(img, i, j, h, w, self.second_size, 
                               self.second_interpolation)
            )
    
    
    def __repr__(self):
        interp_str = (
            ' '.join([
                _pil_interpolation_to_str[interp]
                for interp in self.interpolation
            ]) 
            if isinstance(self.interpolation, (tuple, list))
            else _pil_interpolation_to_str[self.interpolation]
        )

        out = f'{self.__class__.__name__}('
        out += f'size={self.size}, '
        out += f'scale={tuple(round(s, 4) for s in self.scale)}, '
        out += f'ratio={tuple(round(r, 4) for r in self.ratio)}, '
        out += f'interpolation={interp_str}'

        if self.second_size is not None:
            second_interp_str = (
                _pil_interpolation_to_str[self.second_interpolation]
            )

            out += f', second_size={self.second_size}, '
            out += f'second_interpolation={second_interp_str}'
        
        out += ')'

        return out
