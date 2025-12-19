"""
Robust training methods for the DRRL Framework.

Provides SAM optimizer, GroupDRO, and robust loss functions.
"""

from robust.sam import SAM, create_sam_optimizer
from robust.dro import GroupDRO, CVaRDRO, compute_group_losses
from robust.losses import (
    LabelSmoothingCrossEntropy,
    FocalLoss,
    GroupAwareLoss,
    get_loss_function
)


__all__ = [
    'SAM',
    'create_sam_optimizer',
    'GroupDRO',
    'CVaRDRO',
    'compute_group_losses',
    'LabelSmoothingCrossEntropy',
    'FocalLoss',
    'GroupAwareLoss',
    'get_loss_function',
]
