"""
Evaluation metrics for the DRRL Framework.
"""

import torch
import numpy as np
from typing import Dict, List, Optional


def accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor
) -> float:
    """Compute classification accuracy."""
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct / len(targets)


def per_group_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    groups: torch.Tensor,
    n_groups: int
) -> Dict[int, float]:
    """Compute accuracy for each group."""
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    accuracies = {}
    for g in range(n_groups):
        mask = (groups == g)
        if mask.sum() > 0:
            correct = (predictions[mask] == targets[mask]).sum().item()
            accuracies[g] = correct / mask.sum().item()
        else:
            accuracies[g] = 0.0
    
    return accuracies


def worst_group_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    groups: torch.Tensor,
    n_groups: int
) -> float:
    """Compute worst-group accuracy."""
    group_accs = per_group_accuracy(predictions, targets, groups, n_groups)
    return min(group_accs.values()) if group_accs else 0.0


def average_group_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    groups: torch.Tensor,
    n_groups: int
) -> float:
    """Compute average per-group accuracy (macro average)."""
    group_accs = per_group_accuracy(predictions, targets, groups, n_groups)
    if not group_accs:
        return 0.0
    return sum(group_accs.values()) / len(group_accs)


def per_class_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    n_classes: int
) -> Dict[int, float]:
    """Compute per-class accuracy."""
    if predictions.dim() > 1:
        predictions = predictions.argmax(dim=1)
    
    accuracies = {}
    for c in range(n_classes):
        mask = (targets == c)
        if mask.sum() > 0:
            correct = (predictions[mask] == targets[mask]).sum().item()
            accuracies[c] = correct / mask.sum().item()
        else:
            accuracies[c] = 0.0
    
    return accuracies


def ood_gap(
    id_accuracy: float,
    ood_accuracy: float
) -> float:
    """Compute OOD generalization gap."""
    return id_accuracy - ood_accuracy


def robustness_gap(
    average_accuracy: float,
    worst_group_accuracy: float
) -> float:
    """Compute robustness gap between average and worst group."""
    return average_accuracy - worst_group_accuracy


class MetricsTracker:
    """Track and aggregate metrics over batches."""
    
    def __init__(self, n_groups: int, n_classes: int):
        self.n_groups = n_groups
        self.n_classes = n_classes
        self.reset()
    
    def reset(self):
        """Reset all tracking."""
        self.total_correct = 0
        self.total_samples = 0
        self.group_correct = np.zeros(self.n_groups)
        self.group_total = np.zeros(self.n_groups)
        self.class_correct = np.zeros(self.n_classes)
        self.class_total = np.zeros(self.n_classes)
        self.all_predictions = []
        self.all_targets = []
        self.all_groups = []
    
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        groups: torch.Tensor
    ):
        """Update with a batch of predictions."""
        if predictions.dim() > 1:
            predictions = predictions.argmax(dim=1)
        
        predictions = predictions.cpu()
        targets = targets.cpu()
        groups = groups.cpu()
        
        # Overall accuracy
        correct = (predictions == targets)
        self.total_correct += correct.sum().item()
        self.total_samples += len(targets)
        
        # Per-group
        for g in range(self.n_groups):
            mask = (groups == g)
            if mask.sum() > 0:
                self.group_correct[g] += correct[mask].sum().item()
                self.group_total[g] += mask.sum().item()
        
        # Per-class
        for c in range(self.n_classes):
            mask = (targets == c)
            if mask.sum() > 0:
                self.class_correct[c] += correct[mask].sum().item()
                self.class_total[c] += mask.sum().item()
        
        self.all_predictions.extend(predictions.numpy().tolist())
        self.all_targets.extend(targets.numpy().tolist())
        self.all_groups.extend(groups.numpy().tolist())
    
    def compute(self) -> Dict[str, float]:
        """Compute final metrics."""
        metrics = {}
        
        # Overall accuracy
        metrics['accuracy'] = self.total_correct / max(self.total_samples, 1)
        
        # Per-group accuracy
        group_accs = self.group_correct / (self.group_total + 1e-8)
        for g, acc in enumerate(group_accs):
            if self.group_total[g] > 0:
                metrics[f'group_{g}_accuracy'] = acc
        
        metrics['worst_group_accuracy'] = group_accs[self.group_total > 0].min() \
            if (self.group_total > 0).any() else 0.0
        metrics['avg_group_accuracy'] = group_accs[self.group_total > 0].mean() \
            if (self.group_total > 0).any() else 0.0
        
        # Robustness gap
        metrics['robustness_gap'] = metrics['avg_group_accuracy'] - metrics['worst_group_accuracy']
        
        return metrics
