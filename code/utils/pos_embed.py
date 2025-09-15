"""
This module provides positional embedding utilities for 2D sine-cosine encoding and resizing.

Originally adapted from the MAE codebase:
https://github.com/facebookresearch/mae
"""

import torch
import numpy as np


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    Generates 1D sine-cosine positional embeddings from grid positions.

    Args:
        embed_dim (int): Output embedding dimension for each position 
            (must be even).
        pos (np.ndarray): 1D array of positions (shape: (M,)).

    Returns:
        np.ndarray: Positional embeddings of shape (M, embed_dim),
            where each position is encoded using sin and cos functions.
    """
    # Check if embed_dim is even
    assert embed_dim % 2 == 0   # embed_dim = D

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega *= 2.0 / embed_dim
    omega = 1.0 / (10000 ** omega) # (D/2,)

    pos = pos.reshape(-1) # (M,)
    out = np.outer(pos, omega) # (M, D/2)
    emb = np.concatenate([np.sin(out), np.cos(out)], axis=1) # (M, D)

    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    """
    Generates 2D sine-cosine positional embeddings from a mesh grid.

    Args:
        embed_dim (int): Output embedding dimension for each position
            (must be even).
        grid (np.ndarray): Shape (2, 1, H, W) where grid[0] is the
            x-coordinates (W), and grid[1] is the y-coordinates (H).

    Returns:
        np.ndarray: A matrix of shape (H*W, embed_dim) containing 2D
            positional embeddings.
    """
    # Check if embed_dim is even
    assert embed_dim % 2 == 0

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    emb = np.concatenate([emb_h, emb_w], axis=1)

    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    Generates 2D sine-cosine positional embeddings for a square image
    grid.

    Args:
        embed_dim (int): Output embedding dimension for each position
            (must be even).
        grid_size (int): Height and width of the grid.
        cls_token (bool): If True, prepends a zero vector for a
            classification token.

    Returns:
        np.ndarray: Positional embeddings of shape:
            (grid_size**2, embed_dim) if cls_token is False, or
            (1 + grid_size**2, embed_dim) if cls_token is True.
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)

    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)
    grid = grid.reshape(2, 1, grid_size, grid_size)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        cls_embed = np.zeros((1, embed_dim), dtype=np.float32)
        pos_embed = np.concatenate([cls_embed, pos_embed], axis=0)

    return pos_embed


def interpolate_pos_embed(model, checkpoint_model):
    """
    Interpolates pre-trained position embeddings to match model's patch
    resolution.

    Args:
        model (torch.nn.Module): The current Vision Transformer model.
        checkpoint_model (dict): The state_dict loaded from a
            pre-trained checkpoint. Expected to contain a 'pos_embed'
            key.

    Reference:
        https://github.com/facebookresearch/deit
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches

        # Determine grid size of old and new position embeddings
        orig_size = int(
            (pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5
        )
        new_size = int(num_patches ** 0.5)

        if orig_size != new_size:
            print(
                'Positional embeddings interpolate from '
                f'{orig_size}x{orig_size} to {new_size}x{new_size}'
            )

            # Split extra tokens ([CLS], etc.) and position tokens
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

            # Reshape to (1, dim, H_old, W_old) for interpolation
            pos_tokens = pos_tokens.reshape(
                1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)

            # Apply bicubic interpolation to new resolution
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), 
                mode='bicubic', align_corners=False
            )

            # Flatten back to (1, H_new*W_new, dim)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)

            # Concatenate extra tokens back with resized position tokens
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed
