"""
Tests for robust training methods.
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSAM:
    """Tests for SAM optimizer."""
    
    def test_sam_step(self, device):
        """Test SAM optimizer step."""
        from robust import SAM
        
        model = torch.nn.Linear(10, 2).to(device)
        base_optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        sam = SAM(model.parameters(), base_optimizer, rho=0.05)
        
        x = torch.randn(4, 10, device=device)
        y = torch.randint(0, 2, (4,), device=device)
        
        # First step
        loss1 = torch.nn.functional.cross_entropy(model(x), y)
        loss1.backward()
        sam.first_step(zero_grad=True)
        
        # Second step
        loss2 = torch.nn.functional.cross_entropy(model(x), y)
        loss2.backward()
        sam.second_step(zero_grad=True)
    
    def test_create_sam_optimizer(self, device):
        """Test SAM optimizer creation helper."""
        from robust import create_sam_optimizer
        
        model = torch.nn.Linear(10, 2).to(device)
        
        sam = create_sam_optimizer(
            model.parameters(),
            torch.optim.SGD,
            rho=0.05,
            lr=0.01
        )
        
        assert isinstance(sam, SAM)


class TestGroupDRO:
    """Tests for Group DRO."""
    
    def test_group_dro_forward(self, sample_batch, device):
        """Test GroupDRO loss computation."""
        from robust import GroupDRO
        
        images, targets, groups = sample_batch
        targets = targets.to(device)
        groups = groups.to(device)
        
        # Simulate logits
        logits = torch.randn(8, 2, device=device)
        
        dro = GroupDRO(n_groups=4, step_size=0.01).to(device)
        loss = dro(logits, targets, groups)
        
        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
    
    def test_group_weights_update(self, device):
        """Test that group weights get updated."""
        from robust import GroupDRO
        
        dro = GroupDRO(n_groups=4, step_size=0.1).to(device)
        
        initial_weights = dro.get_group_weights().copy()
        
        # Simulate losses
        logits = torch.randn(16, 2, device=device)
        targets = torch.randint(0, 2, (16,), device=device)
        groups = torch.randint(0, 4, (16,), device=device)
        
        _ = dro(logits, targets, groups)
        
        updated_weights = dro.get_group_weights()
        
        # Weights should change
        assert not np.allclose(initial_weights, updated_weights)


class TestRobustLosses:
    """Tests for robust loss functions."""
    
    def test_label_smoothing(self, device):
        """Test label smoothing cross entropy."""
        from robust import LabelSmoothingCrossEntropy
        
        loss_fn = LabelSmoothingCrossEntropy(smoothing=0.1)
        
        logits = torch.randn(4, 10, device=device)
        targets = torch.randint(0, 10, (4,), device=device)
        
        loss = loss_fn(logits, targets)
        
        assert loss.ndim == 0
        assert loss > 0
    
    def test_focal_loss(self, device):
        """Test focal loss."""
        from robust import FocalLoss
        
        loss_fn = FocalLoss(gamma=2.0)
        
        logits = torch.randn(4, 10, device=device)
        targets = torch.randint(0, 10, (4,), device=device)
        
        loss = loss_fn(logits, targets)
        
        assert loss.ndim == 0
        assert loss > 0


# Import numpy for the weights test
import numpy as np
