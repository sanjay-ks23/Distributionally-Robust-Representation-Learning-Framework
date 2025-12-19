"""
Confusion matrix visualization for the DRRL Framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Optional, Dict
from pathlib import Path

from viz.style import set_style, save_figure


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[List[str]] = None,
    normalize: bool = True,
    save_path: Optional[str] = None,
    title: str = 'Confusion Matrix'
) -> plt.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array
        class_names: Names for each class
        normalize: Whether to normalize by row
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Figure object
    """
    set_style('paper')
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt = '.2%'
    else:
        cm_display = cm
        fmt = 'd'
    
    n_classes = cm.shape[0]
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt if not normalize else '.1%',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar=True
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_per_group_confusion(
    group_cms: Dict[int, np.ndarray],
    group_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot confusion matrices for each group.
    
    Args:
        group_cms: Dict mapping group index to confusion matrix
        group_names: Names for each group
        class_names: Names for each class
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    
    n_groups = len(group_cms)
    n_cols = min(n_groups, 2)
    n_rows = (n_groups + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = np.atleast_2d(axes)
    
    for idx, (g, cm) in enumerate(sorted(group_cms.items())):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        
        n_classes = cm.shape[0]
        if class_names is None:
            labels = [f'C{i}' for i in range(n_classes)]
        else:
            labels = class_names
        
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.1%',
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar=False
        )
        
        title = group_names[g] if group_names else f'Group {g}'
        ax.set_title(title)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    # Hide unused axes
    for idx in range(n_groups, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_error_analysis(
    predictions: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    class_names: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Visualize error patterns across groups.
    
    Args:
        predictions: Model predictions
        targets: True labels
        groups: Group labels
        class_names: Names for classes
        group_names: Names for groups
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    
    errors = predictions != targets
    n_groups = len(np.unique(groups))
    n_classes = len(np.unique(targets))
    
    # Error rate by group and class
    error_rates = np.zeros((n_groups, n_classes))
    
    for g in range(n_groups):
        for c in range(n_classes):
            mask = (groups == g) & (targets == c)
            if mask.sum() > 0:
                error_rates[g, c] = errors[mask].mean()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(n_classes)]
    if group_names is None:
        group_names = [f'Group {i}' for i in range(n_groups)]
    
    sns.heatmap(
        error_rates,
        annot=True,
        fmt='.1%',
        cmap='Reds',
        xticklabels=class_names,
        yticklabels=group_names,
        ax=ax
    )
    
    ax.set_xlabel('True Class')
    ax.set_ylabel('Group')
    ax.set_title('Error Rate by Group and Class')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
