"""
Confusion matrix utilities for the DRRL Framework.
"""

import numpy as np
from typing import Optional, Dict


def compute_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_classes: int
) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Returns:
        Matrix of shape (n_classes, n_classes) where [i, j] is count
        of samples with true label i predicted as j.
    """
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    for pred, target in zip(predictions, targets):
        cm[target, pred] += 1
    
    return cm


def normalize_confusion_matrix(
    cm: np.ndarray,
    mode: str = 'true'
) -> np.ndarray:
    """
    Normalize confusion matrix.
    
    Args:
        cm: Confusion matrix
        mode: 'true' (by row), 'pred' (by column), or 'all' (by total)
        
    Returns:
        Normalized confusion matrix
    """
    if mode == 'true':
        return cm / (cm.sum(axis=1, keepdims=True) + 1e-8)
    elif mode == 'pred':
        return cm / (cm.sum(axis=0, keepdims=True) + 1e-8)
    elif mode == 'all':
        return cm / (cm.sum() + 1e-8)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def per_group_confusion_matrix(
    predictions: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_groups: int
) -> Dict[int, np.ndarray]:
    """
    Compute confusion matrix for each group.
    
    Returns:
        Dictionary mapping group index to confusion matrix.
    """
    cms = {}
    
    for g in range(n_groups):
        mask = (groups == g)
        if mask.sum() > 0:
            cms[g] = compute_confusion_matrix(
                predictions[mask],
                targets[mask],
                n_classes
            )
        else:
            cms[g] = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    return cms


def get_classification_report(
    predictions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[list] = None
) -> Dict:
    """
    Generate classification report with precision, recall, F1.
    
    Returns:
        Dictionary with per-class and aggregate metrics.
    """
    n_classes = len(np.unique(targets))
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    
    report = {}
    
    for c in range(n_classes):
        true_c = (targets == c)
        pred_c = (predictions == c)
        
        tp = (true_c & pred_c).sum()
        fp = (~true_c & pred_c).sum()
        fn = (true_c & ~pred_c).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        report[class_names[c]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': true_c.sum()
        }
    
    # Macro average
    report['macro_avg'] = {
        'precision': np.mean([r['precision'] for r in report.values() if 'precision' in r]),
        'recall': np.mean([r['recall'] for r in report.values() if 'recall' in r]),
        'f1': np.mean([r['f1'] for r in report.values() if 'f1' in r])
    }
    
    return report
