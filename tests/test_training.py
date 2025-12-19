"""
Tests for training module.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestTrainers:
    """Tests for trainer classes."""
    
    def test_erm_trainer_init(self, device):
        """Test ERM trainer initialization."""
        from train import ERMTrainer
        from models import DRRLModel
        from torch.utils.data import DataLoader, TensorDataset
        
        model = DRRLModel(encoder='simple_cnn', num_classes=2).to(device)
        
        # Create dummy data
        x = torch.randn(32, 3, 64, 64)
        y = torch.randint(0, 2, (32,))
        g = torch.randint(0, 4, (32,))
        
        dataset = TensorDataset(x, y, g)
        loader = DataLoader(dataset, batch_size=8)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        trainer = ERMTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            device=device,
            epochs=1
        )
        
        assert trainer is not None
    
    def test_trainer_one_epoch(self, device):
        """Test training for one epoch."""
        from train import ERMTrainer
        from models import DRRLModel
        from torch.utils.data import DataLoader, TensorDataset
        
        model = DRRLModel(encoder='simple_cnn', num_classes=2).to(device)
        
        x = torch.randn(16, 3, 64, 64)
        y = torch.randint(0, 2, (16,))
        g = torch.randint(0, 4, (16,))
        
        dataset = TensorDataset(x, y, g)
        loader = DataLoader(dataset, batch_size=8)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        trainer = ERMTrainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            device=device,
            epochs=1
        )
        
        metrics = trainer.train_epoch()
        
        assert 'loss' in metrics
        assert 'accuracy' in metrics


class TestSchedulers:
    """Tests for learning rate schedulers."""
    
    def test_cosine_with_warmup(self, device):
        """Test cosine annealing with warmup."""
        from train import CosineAnnealingWithWarmup
        
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_epochs=5,
            total_epochs=100
        )
        
        # During warmup, LR should increase
        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]['lr'])
            scheduler.step()
        
        # First 5 epochs should show increasing LR
        assert lrs[4] > lrs[0]
    
    def test_get_scheduler(self):
        """Test scheduler factory function."""
        from train import get_scheduler
        
        model = torch.nn.Linear(10, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        scheduler = get_scheduler('cosine', optimizer, total_epochs=100)
        
        assert scheduler is not None


class TestOptimizers:
    """Tests for optimizer utilities."""
    
    def test_get_optimizer(self):
        """Test optimizer factory function."""
        from train import get_optimizer
        
        model = torch.nn.Linear(10, 2)
        
        optimizer = get_optimizer('sgd', model.parameters(), lr=0.01)
        assert isinstance(optimizer, torch.optim.SGD)
        
        optimizer = get_optimizer('adam', model.parameters(), lr=0.01)
        assert isinstance(optimizer, torch.optim.Adam)
