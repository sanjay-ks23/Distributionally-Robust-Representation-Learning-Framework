"""
Training utilities for the DRRL Framework.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from contextlib import contextmanager


@contextmanager
def mixed_precision_context(enabled: bool = True):
    """Context manager for mixed precision training."""
    if enabled and torch.cuda.is_available():
        with torch.cuda.amp.autocast():
            yield
    else:
        yield


def get_optimizer(
    name: str,
    params,
    lr: float,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    **kwargs
) -> torch.optim.Optimizer:
    """Get optimizer by name."""
    if name == 'sgd':
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif name == 'adam':
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def clip_gradients(model: nn.Module, max_norm: float):
    """Clip gradients by global norm."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute total gradient norm."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


class GradScaler:
    """Wrapper for gradient scaling in mixed precision."""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and torch.cuda.is_available()
        if self.enabled:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
    
    def scale(self, loss):
        if self.enabled:
            return self.scaler.scale(loss)
        return loss
    
    def step(self, optimizer):
        if self.enabled:
            self.scaler.step(optimizer)
        else:
            optimizer.step()
    
    def update(self):
        if self.enabled:
            self.scaler.update()
