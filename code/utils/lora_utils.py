"""
This module provides utilities to add and manage LoRA adapters in Vision
Transformer (ViT) models for efficient pretraining.
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    Wraps an nn.Linear with LoRA adapters.
    
    Forward: y = x W^T + s * x A B^T, where s = alpha / r.
    Only A and B are trainable in the adapter phase.
    """
    def __init__(
        self,
        base_linear: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        train_bias: bool = True # BitFit by default
    ):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)
        assert r > 0

        w = base_linear.weight
        device, dtype = w.device, w.dtype

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        self.r = r
        self.scaling = alpha / r

        self.weight = nn.Parameter(
            w.detach().clone(), requires_grad=False
        )
        self.bias = None
        if base_linear.bias is not None:
            self.bias = nn.Parameter(
                base_linear.bias.detach().clone(), requires_grad=train_bias
            )

        self.lora_A = nn.Parameter(torch.zeros(
            self.in_features, r, device=device, dtype=dtype
        ))
        self.lora_B = nn.Parameter(torch.zeros(
            self.out_features, r, device=device, dtype=dtype
        ))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = nn.functional.linear(x, self.weight, self.bias)
        lora = (x @ self.lora_A) @ self.lora_B.t()

        return base + self.dropout(lora) * self.scaling
    

    @torch.no_grad()
    def fuse_into_base(self) -> None:
        # W <- W + s * (B @ A^T)
        update = (self.lora_B @ self.lora_A.t()) * self.scaling
        self.weight += update
        self.lora_A.zero_()
        self.lora_B.zero_()


def _warp_linear(
    module: nn.Module,
    name: str,
    r: int,
    alpha: int,
    dropout: float
) -> None:
    lin = getattr(module, name, None)
    if isinstance(lin, nn.Linear):
        setattr(
            module,
            name,
            LoRALinear(lin, r=r, alpha=alpha, dropout=dropout)
        )


def add_lora_to_vit(
    model: nn.Module,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.0,
) -> nn.Module:
    """
    Attaches LoRA to common ViT submodules:
    - attention: qkv (or q, k, v), proj
    - MLP: fc1, fc2
    """
    for m in model.modules():
        if hasattr(m, 'attn'):
            if hasattr(m.attn, 'qkv'):
                _warp_linear(m.attn, 'qkv', r, alpha, dropout)
            else:
                _warp_linear(m.attn, 'q', r, alpha, dropout)
                _warp_linear(m.attn, 'k', r, alpha, dropout)
                _warp_linear(m.attn, 'v', r, alpha, dropout)
            if hasattr(m.attn, 'proj'):
                _warp_linear(m.attn, 'proj', r, alpha, dropout)
        
        if hasattr(m, 'mlp'):
            _warp_linear(m.mlp, 'fc1', r, alpha, dropout)
            _warp_linear(m.mlp, 'fc2', r, alpha, dropout)

    return model


def set_trainable_for_adapter_phase(model: nn.Module) -> None:
    # Freeze all parameters first
    for _, param in model.named_parameters():
        param.requires_grad = False

    lightweight_allowlist = [
        'pos_embed',
        'patch_embed',
        'cls_token',
        'mask_token',
        'decoder_pos_embed',
        'decoder_embed',
        'decoder_pred',
        'decoder_norm'
    ]

    # Enable LoRA factors + biases + LayerNorm + lightweight params
    for name, param in model.named_parameters():
        if '.lora_A' in name or '.lora_B' in name or name.endswith('.bias'):
            param.requires_grad = True
        elif any(tag in name for tag in lightweight_allowlist):
            param.requires_grad = True

    for m in model.modules():
        if isinstance(m, nn.LayerNorm):
            for param in m.parameters():
                param.requires_grad = True


def fuse_adapters(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.fuse_into_base()
