"""
Main DRRL model combining encoder and classifier.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from models.encoders import get_encoder, ENCODER_DIMS
from models.classifiers import get_classifier


class DRRLModel(nn.Module):
    """
    Main model for Distributionally Robust Representation Learning.
    
    Combines an encoder (feature extractor) with a classifier head.
    Supports returning embeddings for visualization and analysis.
    
    Example:
        >>> model = DRRLModel(
        ...     encoder='resnet18',
        ...     num_classes=2,
        ...     classifier_type='linear'
        ... )
        >>> logits, embeddings = model(images, return_embeddings=True)
    """
    
    def __init__(
        self,
        encoder: str = 'resnet18',
        num_classes: int = 2,
        classifier_type: str = 'linear',
        pretrained: bool = True,
        freeze_encoder: bool = False,
        embedding_dim: Optional[int] = None,
        hidden_dims: list = [256],
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.encoder_name = encoder
        self.num_classes = num_classes
        
        # Get embedding dimension
        if embedding_dim is None:
            embedding_dim = ENCODER_DIMS.get(encoder, 512)
        
        # Build encoder
        self.encoder = get_encoder(
            name=encoder,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            freeze=freeze_encoder
        )
        
        # Build classifier
        self.classifier = get_classifier(
            classifier_type=classifier_type,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, height, width)
            return_embeddings: Whether to return embeddings
            
        Returns:
            Tuple of (logits, embeddings) if return_embeddings else (logits, None)
        """
        embeddings = self.encoder(x)
        logits = self.classifier(embeddings)
        
        if return_embeddings:
            return logits, embeddings
        return logits, None
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract embeddings without classification."""
        return self.encoder(x)
    
    def freeze_encoder(self) -> None:
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self) -> None:
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True


def build_model(config: Dict[str, Any]) -> DRRLModel:
    """
    Build model from configuration dictionary.
    
    Args:
        config: Model configuration with keys:
            - encoder: Encoder name
            - num_classes: Number of classes
            - classifier_type: Classifier type
            - pretrained: Use pretrained weights
            - etc.
            
    Returns:
        DRRLModel instance
    """
    return DRRLModel(
        encoder=config.get('encoder', 'resnet18'),
        num_classes=config.get('num_classes', 2),
        classifier_type=config.get('classifier_type', 'linear'),
        pretrained=config.get('pretrained', True),
        freeze_encoder=config.get('freeze_encoder', False),
        embedding_dim=config.get('embedding_dim'),
        hidden_dims=config.get('hidden_dims', [256]),
        dropout=config.get('dropout', 0.5)
    )
