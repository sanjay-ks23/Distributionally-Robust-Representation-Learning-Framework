"""
Tests for the models module.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestEncoders:
    """Tests for encoder architectures."""
    
    def test_simple_cnn_encoder(self, device):
        """Test SimpleCNN encoder."""
        from models import SimpleCNNEncoder
        
        encoder = SimpleCNNEncoder(
            in_channels=3,
            embedding_dim=128,
            input_size=64
        ).to(device)
        
        x = torch.randn(4, 3, 64, 64, device=device)
        output = encoder(x)
        
        assert output.shape == (4, 128)
    
    def test_resnet_encoder(self, device):
        """Test ResNet encoder."""
        from models import ResNetEncoder
        
        encoder = ResNetEncoder(
            arch='resnet18',
            pretrained=False
        ).to(device)
        
        x = torch.randn(2, 3, 224, 224, device=device)
        output = encoder(x)
        
        assert output.shape == (2, 512)


class TestClassifiers:
    """Tests for classifier heads."""
    
    def test_linear_classifier(self, device):
        """Test linear classifier."""
        from models import LinearClassifier
        
        classifier = LinearClassifier(
            embedding_dim=128,
            num_classes=2
        ).to(device)
        
        x = torch.randn(4, 128, device=device)
        output = classifier(x)
        
        assert output.shape == (4, 2)
    
    def test_mlp_classifier(self, device):
        """Test MLP classifier."""
        from models import MLPClassifier
        
        classifier = MLPClassifier(
            embedding_dim=128,
            num_classes=2,
            hidden_dims=[64, 32],
            dropout=0.5
        ).to(device)
        
        x = torch.randn(4, 128, device=device)
        output = classifier(x)
        
        assert output.shape == (4, 2)


class TestDRRLModel:
    """Tests for the main DRRL model."""
    
    def test_model_forward(self, sample_batch, device):
        """Test model forward pass."""
        from models import DRRLModel
        
        model = DRRLModel(
            encoder='simple_cnn',
            num_classes=2
        ).to(device)
        
        images, targets, groups = sample_batch
        images = images.to(device)
        
        logits, embeddings = model(images, return_embeddings=True)
        
        assert logits.shape == (8, 2)
        assert embeddings.shape[0] == 8
    
    def test_model_build_from_config(self, model_config, device):
        """Test building model from config."""
        from models import build_model
        
        model = build_model(model_config).to(device)
        
        x = torch.randn(4, 3, 64, 64, device=device)
        logits, _ = model(x)
        
        assert logits.shape == (4, 2)
    
    def test_get_embeddings(self, device):
        """Test embedding extraction."""
        from models import DRRLModel
        
        model = DRRLModel(encoder='simple_cnn', num_classes=2).to(device)
        
        x = torch.randn(4, 3, 64, 64, device=device)
        embeddings = model.get_embeddings(x)
        
        assert embeddings.shape[0] == 4
        assert embeddings.shape[1] == model.embedding_dim
