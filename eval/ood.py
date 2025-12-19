"""
OOD (Out-of-Distribution) evaluation utilities.
"""

import numpy as np
from typing import Dict, List


def compute_ood_metrics(
    id_predictions: np.ndarray,
    id_targets: np.ndarray,
    ood_predictions: np.ndarray,
    ood_targets: np.ndarray
) -> Dict[str, float]:
    """
    Compute OOD generalization metrics.
    
    Args:
        id_predictions: In-distribution predictions
        id_targets: In-distribution targets
        ood_predictions: OOD predictions
        ood_targets: OOD targets
        
    Returns:
        Dictionary of OOD metrics
    """
    id_acc = (id_predictions == id_targets).mean()
    ood_acc = (ood_predictions == ood_targets).mean()
    
    return {
        'id_accuracy': id_acc,
        'ood_accuracy': ood_acc,
        'ood_gap': id_acc - ood_acc,
        'relative_drop': (id_acc - ood_acc) / (id_acc + 1e-8)
    }


def compute_corruption_robustness(
    clean_accuracy: float,
    corruption_accuracies: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute corruption robustness metrics.
    
    Args:
        clean_accuracy: Accuracy on clean data
        corruption_accuracies: Dict mapping corruption name to accuracy
        
    Returns:
        Robustness metrics
    """
    mean_corrupt_acc = np.mean(list(corruption_accuracies.values()))
    
    return {
        'clean_accuracy': clean_accuracy,
        'mean_corruption_accuracy': mean_corrupt_acc,
        'mean_corruption_error': 1 - mean_corrupt_acc,
        'relative_robustness': mean_corrupt_acc / (clean_accuracy + 1e-8),
        'corruption_gap': clean_accuracy - mean_corrupt_acc
    }


def effective_robustness(
    id_accuracy: float,
    ood_accuracy: float,
    baseline_id: float,
    baseline_ood: float
) -> float:
    """
    Compute effective robustness.
    
    Measures how much better the model is at OOD generalization
    relative to what would be expected from its ID performance.
    """
    # Expected OOD based on linear relation
    expected_ood = baseline_ood + (id_accuracy - baseline_id)
    
    # Effective robustness = actual OOD - expected OOD
    return ood_accuracy - expected_ood


class OODEvaluator:
    """Evaluate model on multiple OOD datasets."""
    
    def __init__(self, evaluator, id_dataloader):
        self.evaluator = evaluator
        self.id_dataloader = id_dataloader
        self.id_results = None
    
    def evaluate_id(self):
        """Evaluate on in-distribution data."""
        self.id_results = self.evaluator.evaluate(self.id_dataloader)
        return self.id_results
    
    def evaluate_ood(self, ood_dataloader, name: str = 'ood') -> Dict:
        """Evaluate on OOD data and compute gap."""
        if self.id_results is None:
            self.evaluate_id()
        
        ood_results = self.evaluator.evaluate(ood_dataloader)
        
        return {
            f'{name}_accuracy': ood_results['accuracy'],
            f'{name}_gap': self.id_results['accuracy'] - ood_results['accuracy'],
            f'{name}_worst_group_gap': (
                self.id_results['worst_group_accuracy'] - 
                ood_results['worst_group_accuracy']
            )
        }
