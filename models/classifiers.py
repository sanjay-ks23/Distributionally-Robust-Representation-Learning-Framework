"""
Classifier heads for the DRRL Framework.

Classification layers that map embeddings to class logits.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class LinearClassifier(nn.Module):
    """Simple linear classifier head."""
    
    def __init__(self, embedding_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class MLPClassifier(nn.Module):
    """
    Multi-layer perceptron classifier.
    
    Architecture: Linear -> ReLU -> Dropout -> ... -> Linear
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        hidden_dims: List[int] = [256],
        dropout: float = 0.5,
        use_batchnorm: bool = True
    ):
        super().__init__()
        
        layers = []
        prev_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class CosineClassifier(nn.Module):
    """
    Cosine similarity-based classifier.
    
    Computes cosine similarity between embeddings and learned class prototypes.
    """
    
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        temperature: float = 0.1
    ):
        super().__init__()
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.prototypes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize embeddings and prototypes
        x_norm = nn.functional.normalize(x, dim=1)
        p_norm = nn.functional.normalize(self.prototypes, dim=1)
        
        # Compute cosine similarity
        logits = torch.mm(x_norm, p_norm.t()) / self.temperature
        return logits


def get_classifier(
    classifier_type: str,
    embedding_dim: int,
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function for classifiers.
    
    Args:
        classifier_type: 'linear', 'mlp', or 'cosine'
        embedding_dim: Input embedding dimension
        num_classes: Number of output classes
        
    Returns:
        Classifier module
    """
    if classifier_type == 'linear':
        return LinearClassifier(embedding_dim, num_classes)
    elif classifier_type == 'mlp':
        return MLPClassifier(embedding_dim, num_classes, **kwargs)
    elif classifier_type == 'cosine':
        return CosineClassifier(embedding_dim, num_classes, **kwargs)
    else:
        raise ValueError(f"Unknown classifier: {classifier_type}")
