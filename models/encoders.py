"""
Encoder architectures for the DRRL Framework.

Feature extractors that produce embeddings from input data.
"""

import torch
import torch.nn as nn
from torchvision import models
from typing import Optional


class SimpleCNNEncoder(nn.Module):
    """
    Simple CNN encoder for synthetic/small datasets.
    
    Architecture:
        Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> Flatten
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        embedding_dim: int = 128,
        input_size: int = 64
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        self.fc = nn.Linear(256, embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetEncoder(nn.Module):
    """
    ResNet-based encoder using pretrained weights.
    
    Removes the final FC layer and optionally adds a projection head.
    """
    
    def __init__(
        self,
        arch: str = 'resnet18',
        pretrained: bool = True,
        embedding_dim: Optional[int] = None,
        freeze: bool = False
    ):
        super().__init__()
        self.arch = arch
        
        # Load pretrained ResNet
        if arch == 'resnet18':
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif arch == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        
        # Remove final FC layer
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])
        
        # Optional projection head
        if embedding_dim is not None and embedding_dim != self.feature_dim:
            self.projection = nn.Linear(self.feature_dim, embedding_dim)
            self.embedding_dim = embedding_dim
        else:
            self.projection = None
            self.embedding_dim = self.feature_dim
        
        # Freeze encoder if specified
        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        if self.projection is not None:
            x = self.projection(x)
        return x


def get_encoder(
    name: str,
    pretrained: bool = True,
    embedding_dim: Optional[int] = None,
    freeze: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function for encoders.
    
    Args:
        name: Encoder name ('resnet18', 'resnet50', 'simple_cnn')
        pretrained: Use pretrained weights (for ResNet)
        embedding_dim: Output embedding dimension
        freeze: Freeze encoder weights
        
    Returns:
        Encoder module
    """
    if name == 'simple_cnn':
        return SimpleCNNEncoder(
            embedding_dim=embedding_dim or 128,
            **kwargs
        )
    elif name in ['resnet18', 'resnet50']:
        return ResNetEncoder(
            arch=name,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            freeze=freeze
        )
    else:
        raise ValueError(f"Unknown encoder: {name}")


# Embedding dimensions for each encoder
ENCODER_DIMS = {
    'simple_cnn': 128,
    'resnet18': 512,
    'resnet50': 2048,
}
