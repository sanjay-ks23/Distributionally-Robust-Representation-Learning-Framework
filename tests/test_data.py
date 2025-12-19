"""
Tests for the data module.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSyntheticDataset:
    """Tests for SyntheticSpuriousDataset."""
    
    def test_dataset_creation(self, seed):
        """Test dataset creation and basic properties."""
        from data import SyntheticSpuriousDataset
        
        dataset = SyntheticSpuriousDataset(
            n_samples=100,
            spurious_correlation=0.9,
            image_size=32,
            seed=seed
        )
        dataset.load_data()
        
        assert len(dataset) == 100
        assert dataset.metadata.num_classes == 2
        assert dataset.metadata.num_groups == 4
    
    def test_dataset_getitem(self, synthetic_dataset):
        """Test __getitem__ method."""
        x, y, g = synthetic_dataset[0]
        
        assert isinstance(x, torch.Tensor)
        assert x.shape == (3, 32, 32)
        assert isinstance(y, int)
        assert isinstance(g, int)
        assert 0 <= y < 2
        assert 0 <= g < 4
    
    def test_group_counts(self, synthetic_dataset):
        """Test group count computation."""
        counts = synthetic_dataset.get_group_counts()
        
        assert len(counts) == 4
        assert counts.sum() == len(synthetic_dataset)
    
    def test_spurious_correlation(self, seed):
        """Test that spurious correlation is approximately correct."""
        from data import SyntheticSpuriousDataset
        
        dataset = SyntheticSpuriousDataset(
            n_samples=10000,
            spurious_correlation=0.9,
            seed=seed
        )
        dataset.load_data()
        
        targets = dataset.targets.numpy()
        groups = dataset.groups.numpy()
        
        # Group 0: class 0, spurious 0 (majority)
        # Group 3: class 1, spurious 1 (majority)
        majority_groups = np.isin(groups, [0, 3])
        correlation = majority_groups.mean()
        
        assert 0.85 < correlation < 0.95


class TestDataLoaders:
    """Tests for data loaders."""
    
    def test_create_dataloaders(self, synthetic_dataset):
        """Test dataloader creation."""
        from data import create_dataloaders
        
        train_loader, val_loader, test_loader = create_dataloaders(
            synthetic_dataset,
            synthetic_dataset,
            synthetic_dataset,
            batch_size=16
        )
        
        batch = next(iter(train_loader))
        assert len(batch) == 3
        assert batch[0].shape[0] == 16


class TestTransforms:
    """Tests for data transforms."""
    
    def test_train_transform(self):
        """Test training transform."""
        from data import get_train_transform
        
        transform = get_train_transform(image_size=64)
        assert transform is not None
    
    def test_eval_transform(self):
        """Test evaluation transform."""
        from data import get_eval_transform
        
        transform = get_eval_transform(image_size=64)
        assert transform is not None
