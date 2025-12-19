"""
Group Distributionally Robust Optimization (DRO).

Optimizes worst-group performance by dynamically reweighting groups.

Reference: Sagawa et al. "Distributionally Robust Neural Networks for
           Group Shifts" (ICLR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


class GroupDRO(nn.Module):
    """
    Group Distributionally Robust Optimization loss.
    
    Maintains group weights that are adjusted based on per-group losses.
    Groups with higher losses receive higher weights, focusing learning
    on the worst-performing groups.
    
    Example:
        >>> dro = GroupDRO(n_groups=4, step_size=0.01)
        >>> 
        >>> # Training step
        >>> logits = model(inputs)
        >>> loss = dro(logits, targets, groups)
        >>> loss.backward()
    """
    
    def __init__(
        self,
        n_groups: int,
        step_size: float = 0.01,
        normalize_loss: bool = True,
        gamma: float = 0.1,
        alpha: float = 0.0,
        use_exp_weights: bool = True
    ):
        """
        Initialize GroupDRO.
        
        Args:
            n_groups: Number of groups in the dataset.
            step_size: Learning rate for group weight updates.
            normalize_loss: Normalize losses by group counts.
            gamma: Adjustment factor for robust training.
            alpha: Generalization adjustment term.
            use_exp_weights: Use exponentiated gradient for weights.
        """
        super().__init__()
        
        self.n_groups = n_groups
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.gamma = gamma
        self.alpha = alpha
        self.use_exp_weights = use_exp_weights
        
        # Initialize uniform group weights
        self.register_buffer(
            'group_weights',
            torch.ones(n_groups) / n_groups
        )
        
        # Track group loss history for logging
        self.register_buffer(
            'group_loss_history',
            torch.zeros(n_groups)
        )
        
        # Counts for this epoch
        self.register_buffer(
            'group_counts',
            torch.zeros(n_groups)
        )
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        groups: torch.Tensor,
        update_weights: bool = True
    ) -> torch.Tensor:
        """
        Compute DRO loss.
        
        Args:
            logits: Model outputs of shape (batch_size, num_classes)
            targets: Target labels of shape (batch_size,)
            groups: Group labels of shape (batch_size,)
            update_weights: Whether to update group weights
            
        Returns:
            Weighted loss scalar
        """
        device = logits.device
        batch_size = logits.size(0)
        
        # Compute per-sample losses
        per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Compute per-group losses
        group_losses = torch.zeros(self.n_groups, device=device)
        group_counts = torch.zeros(self.n_groups, device=device)
        
        for g in range(self.n_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                group_losses[g] = per_sample_loss[mask].mean()
                group_counts[g] = mask.sum().float()
        
        # Update tracking buffers
        self.group_loss_history = group_losses.detach()
        self.group_counts = group_counts
        
        # Update group weights
        if update_weights:
            self._update_weights(group_losses)
        
        # Compute weighted loss
        if self.normalize_loss:
            # Weight by group weights, accounting for group sizes
            weighted_loss = (self.group_weights * group_losses).sum()
        else:
            # Direct weighted combination
            weighted_loss = (self.group_weights * group_losses).sum()
        
        return weighted_loss
    
    @torch.no_grad()
    def _update_weights(self, group_losses: torch.Tensor) -> None:
        """Update group weights based on losses."""
        if self.use_exp_weights:
            # Exponentiated gradient descent
            adjusted_losses = group_losses + self.gamma
            self.group_weights = self.group_weights * torch.exp(
                self.step_size * adjusted_losses
            )
            # Normalize to sum to 1
            self.group_weights = self.group_weights / self.group_weights.sum()
        else:
            # Direct weight update
            self.group_weights = self.group_weights + self.step_size * group_losses
            self.group_weights = F.softmax(self.group_weights, dim=0)
    
    def get_group_weights(self) -> np.ndarray:
        """Get current group weights."""
        return self.group_weights.cpu().numpy()
    
    def get_group_losses(self) -> np.ndarray:
        """Get latest per-group losses."""
        return self.group_loss_history.cpu().numpy()
    
    def reset_weights(self) -> None:
        """Reset weights to uniform distribution."""
        self.group_weights.fill_(1.0 / self.n_groups)


class CVaRDRO(nn.Module):
    """
    Conditional Value at Risk (CVaR) DRO.
    
    Focuses on the worst alpha-fraction of samples.
    """
    
    def __init__(self, alpha: float = 0.2):
        """
        Initialize CVaR-DRO.
        
        Args:
            alpha: Fraction of worst samples to focus on (0 < alpha <= 1).
        """
        super().__init__()
        assert 0 < alpha <= 1.0, f"alpha must be in (0, 1], got {alpha}"
        self.alpha = alpha
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        groups: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute CVaR loss.
        
        Args:
            logits: Model outputs
            targets: Target labels
            groups: Unused, for API compatibility
            
        Returns:
            CVaR loss
        """
        # Compute per-sample losses
        per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Sort losses in descending order
        sorted_losses, _ = torch.sort(per_sample_loss, descending=True)
        
        # Take the worst alpha fraction
        n_samples = int(len(sorted_losses) * self.alpha)
        n_samples = max(1, n_samples)
        
        cvar_loss = sorted_losses[:n_samples].mean()
        return cvar_loss


def compute_group_losses(
    logits: torch.Tensor,
    targets: torch.Tensor,
    groups: torch.Tensor,
    n_groups: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-group losses.
    
    Args:
        logits: Model outputs
        targets: Target labels
        groups: Group labels
        n_groups: Number of groups
        
    Returns:
        Tuple of (group_losses, group_counts)
    """
    device = logits.device
    per_sample_loss = F.cross_entropy(logits, targets, reduction='none')
    
    group_losses = torch.zeros(n_groups, device=device)
    group_counts = torch.zeros(n_groups, device=device)
    
    for g in range(n_groups):
        mask = (groups == g)
        if mask.sum() > 0:
            group_losses[g] = per_sample_loss[mask].mean()
            group_counts[g] = mask.sum().float()
    
    return group_losses, group_counts
