"""
Training module for the DRRL Framework.
"""

from train.base_trainer import BaseTrainer
from train.erm_trainer import ERMTrainer
from train.sam_trainer import SAMTrainer
from train.dro_trainer import DROTrainer
from train.schedulers import (
    CosineAnnealingWithWarmup,
    LinearWarmupScheduler,
    get_scheduler
)
from train.utils import get_optimizer, clip_gradients, GradScaler


TRAINER_REGISTRY = {
    'erm': ERMTrainer,
    'sam': SAMTrainer,
    'dro': DROTrainer,
}


def get_trainer(method: str, **kwargs):
    """Get trainer by training method."""
    if method not in TRAINER_REGISTRY:
        raise ValueError(f"Unknown method: {method}")
    return TRAINER_REGISTRY[method](**kwargs)


__all__ = [
    'BaseTrainer',
    'ERMTrainer',
    'SAMTrainer',
    'DROTrainer',
    'get_trainer',
    'get_scheduler',
    'get_optimizer',
    'CosineAnnealingWithWarmup',
]
