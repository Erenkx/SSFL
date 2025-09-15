"""
Layer-wise Adaptive Rate Scaling (LARS) optimizer.

Originally adapted from the MAE codebase:
https://github.com/facebookresearch/mae
"""

import torch


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer.

    No rate scaling or weight decay for parameters of dim <= 1.

    Reference:
        https://github.com/facebookresearch/moco-v3
    """
    def __init__(
        self,
        params,
        lr=0,
        weight_decay=0,
        momentum=0.9,
        trust_coefficient=0.001
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            trust_coefficient=trust_coefficient
        )
        super().__init__(params, defaults)


    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for param in group['params']:
                grad = param.grad

                if grad is None:
                    continue

                if param.ndim > 1:
                    grad = grad.add(param, alpha=group['weight_decay'])

                    param_norm = torch.norm(param)
                    grad_norm = torch.norm(grad)
                    one = torch.ones_like(param_norm)

                    trust_ratio = torch.where(
                        param_norm > 0.0,
                        torch.where(
                            grad_norm > 0,
                            (
                                group['trust_coefficient']
                                * param_norm
                                / grad_norm
                            ),
                            one
                        ),
                        one
                    )
                    grad = grad.mul(trust_ratio)
    
                state = self.state[param]
                if 'mu' not in state:
                    state['mu'] = torch.zeros_like(param)
                    
                mu = state['mu']
                mu.mul_(group['momentum']).add_(grad)

                param.add_(mu, alpha=-group['lr'])
