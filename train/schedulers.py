"""
Learning rate schedulers for the DRRL Framework.
"""

import math
import torch
from torch.optim.lr_scheduler import _LRScheduler
from typing import Optional


class CosineAnnealingWithWarmup(_LRScheduler):
    """Cosine annealing scheduler with linear warmup."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            return [
                self.min_lr + (base_lr - self.min_lr) * 
                (1 + math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class LinearWarmupScheduler(_LRScheduler):
    """Linear warmup followed by constant learning rate."""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        return self.base_lrs


def get_scheduler(
    name: str,
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    **kwargs
) -> Optional[_LRScheduler]:
    """Get scheduler by name."""
    if name == 'cosine':
        return CosineAnnealingWithWarmup(
            optimizer, warmup_epochs, total_epochs, min_lr
        )
    elif name == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=kwargs.get('step_size', 20), gamma=0.1
        )
    elif name == 'warmup':
        return LinearWarmupScheduler(optimizer, warmup_epochs)
    elif name == 'none' or name is None:
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")
