"""
Evaluation module for the DRRL Framework.
"""

from eval.metrics import (
    accuracy,
    per_group_accuracy,
    worst_group_accuracy,
    average_group_accuracy,
    per_class_accuracy,
    ood_gap,
    robustness_gap,
    MetricsTracker
)
from eval.evaluator import Evaluator, compare_models
from eval.confusion import (
    compute_confusion_matrix,
    normalize_confusion_matrix,
    per_group_confusion_matrix,
    get_classification_report
)
from eval.ood import (
    compute_ood_metrics,
    compute_corruption_robustness,
    OODEvaluator
)


__all__ = [
    'accuracy',
    'per_group_accuracy',
    'worst_group_accuracy',
    'average_group_accuracy',
    'MetricsTracker',
    'Evaluator',
    'compare_models',
    'compute_confusion_matrix',
    'normalize_confusion_matrix',
    'per_group_confusion_matrix',
    'compute_ood_metrics',
    'OODEvaluator',
]
