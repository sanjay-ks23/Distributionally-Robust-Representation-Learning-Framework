"""
Models module for the DRRL Framework.

Provides encoder architectures, classifier heads, and the main DRRL model.
"""

from models.encoders import (
    SimpleCNNEncoder,
    ResNetEncoder,
    get_encoder,
    ENCODER_DIMS
)
from models.classifiers import (
    LinearClassifier,
    MLPClassifier,
    CosineClassifier,
    get_classifier
)
from models.drrl_model import DRRLModel, build_model


__all__ = [
    'SimpleCNNEncoder',
    'ResNetEncoder',
    'get_encoder',
    'ENCODER_DIMS',
    'LinearClassifier',
    'MLPClassifier',
    'CosineClassifier',
    'get_classifier',
    'DRRLModel',
    'build_model',
]
