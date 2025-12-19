"""
Robust loss functions for the DRRL Framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy with label smoothing."""
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        n_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_targets * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class GroupAwareLoss(nn.Module):
    """Loss that tracks per-group statistics."""
    
    def __init__(self, n_groups: int, base_loss: str = 'ce'):
        super().__init__()
        self.n_groups = n_groups
        
        if base_loss == 'ce':
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        elif base_loss == 'focal':
            self.loss_fn = FocalLoss(reduction='none')
        else:
            raise ValueError(f"Unknown loss: {base_loss}")
        
        self.register_buffer('group_losses', torch.zeros(n_groups))
        self.register_buffer('group_counts', torch.zeros(n_groups))
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        groups: torch.Tensor
    ) -> torch.Tensor:
        per_sample_loss = self.loss_fn(logits, targets)
        
        # Track per-group losses
        self.group_losses.zero_()
        self.group_counts.zero_()
        
        for g in range(self.n_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                self.group_losses[g] = per_sample_loss[mask].mean()
                self.group_counts[g] = mask.sum()
        
        return per_sample_loss.mean()


def get_loss_function(
    name: str,
    n_groups: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """Get loss function by name."""
    if name == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    elif name == 'label_smoothing':
        return LabelSmoothingCrossEntropy(**kwargs)
    elif name == 'focal':
        return FocalLoss(**kwargs)
    elif name == 'group_aware':
        return GroupAwareLoss(n_groups=n_groups, **kwargs)
    else:
        raise ValueError(f"Unknown loss: {name}")
