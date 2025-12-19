"""
Tests for the evaluation module.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMetrics:
    """Tests for evaluation metrics."""
    
    def test_accuracy(self):
        """Test accuracy computation."""
        from eval import accuracy
        
        preds = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])
        
        acc = accuracy(preds, targets)
        
        assert acc == 0.8
    
    def test_per_group_accuracy(self):
        """Test per-group accuracy."""
        from eval import per_group_accuracy
        
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 1, 1])
        groups = torch.tensor([0, 0, 1, 1])
        
        group_accs = per_group_accuracy(preds, targets, groups, n_groups=2)
        
        assert group_accs[0] == 1.0  # Both correct in group 0
        assert group_accs[1] == 0.5  # 1/2 correct in group 1
    
    def test_worst_group_accuracy(self):
        """Test worst-group accuracy."""
        from eval import worst_group_accuracy
        
        preds = torch.tensor([0, 1, 0, 1])
        targets = torch.tensor([0, 1, 1, 1])
        groups = torch.tensor([0, 0, 1, 1])
        
        worst_acc = worst_group_accuracy(preds, targets, groups, n_groups=2)
        
        assert worst_acc == 0.5  # Group 1 has 50%


class TestConfusionMatrix:
    """Tests for confusion matrix utilities."""
    
    def test_compute_confusion_matrix(self):
        """Test confusion matrix computation."""
        from eval import compute_confusion_matrix
        
        preds = np.array([0, 1, 0, 1, 1])
        targets = np.array([0, 1, 1, 0, 1])
        
        cm = compute_confusion_matrix(preds, targets, n_classes=2)
        
        assert cm.shape == (2, 2)
        assert cm.sum() == 5
    
    def test_normalize_confusion_matrix(self):
        """Test confusion matrix normalization."""
        from eval import normalize_confusion_matrix
        
        cm = np.array([[10, 2], [1, 8]])
        
        cm_norm = normalize_confusion_matrix(cm, mode='true')
        
        # Each row should sum to 1
        assert np.allclose(cm_norm.sum(axis=1), 1.0)


class TestMetricsTracker:
    """Tests for MetricsTracker."""
    
    def test_metrics_tracker(self):
        """Test metrics tracker update and compute."""
        from eval import MetricsTracker
        
        tracker = MetricsTracker(n_groups=4, n_classes=2)
        
        # Simulate a batch
        preds = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 0, 1, 1, 1])
        groups = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
        
        tracker.update(preds, targets, groups)
        
        metrics = tracker.compute()
        
        assert 'accuracy' in metrics
        assert 'worst_group_accuracy' in metrics
        assert 0 <= metrics['accuracy'] <= 1
