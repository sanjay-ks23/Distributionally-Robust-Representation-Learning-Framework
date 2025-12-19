"""
Pytest fixtures for DRRL Framework tests.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def device():
    """Get test device."""
    return torch.device('cpu')


@pytest.fixture
def seed():
    """Set random seed for reproducibility."""
    torch.manual_seed(42)
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    batch_size = 8
    channels = 3
    size = 64
    n_classes = 2
    n_groups = 4
    
    images = torch.randn(batch_size, channels, size, size)
    targets = torch.randint(0, n_classes, (batch_size,))
    groups = torch.randint(0, n_groups, (batch_size,))
    
    return images, targets, groups


@pytest.fixture
def model_config():
    """Sample model configuration."""
    return {
        'encoder': 'simple_cnn',
        'num_classes': 2,
        'classifier_type': 'linear',
        'pretrained': False,
        'embedding_dim': 128
    }


@pytest.fixture
def synthetic_dataset():
    """Create a small synthetic dataset for testing."""
    from data import SyntheticSpuriousDataset
    
    dataset = SyntheticSpuriousDataset(
        n_samples=100,
        spurious_correlation=0.9,
        image_size=32,
        seed=42
    )
    dataset.load_data()
    return dataset
