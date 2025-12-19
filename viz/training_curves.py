"""
Training curves visualization for the DRRL Framework.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from viz.style import DRRL_COLORS, set_style, save_figure, get_figure


def plot_training_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = 'Training Curves'
) -> plt.Figure:
    """
    Plot training loss and accuracy curves.
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', etc.
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history.get('train_loss', [])) + 1)
    
    # Loss plot
    ax = axes[0]
    if 'train_loss' in history:
        ax.plot(epochs, history['train_loss'], label='Train', color=DRRL_COLORS['primary'])
    if 'val_loss' in history:
        ax.plot(epochs, history['val_loss'], label='Validation', color=DRRL_COLORS['secondary'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax = axes[1]
    if 'train_accuracy' in history:
        ax.plot(epochs, history['train_accuracy'], label='Train', color=DRRL_COLORS['primary'])
    if 'val_accuracy' in history:
        ax.plot(epochs, history['val_accuracy'], label='Validation', color=DRRL_COLORS['secondary'])
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_method_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    metric: str = 'val_accuracy',
    save_path: Optional[str] = None,
    title: str = 'Method Comparison'
) -> plt.Figure:
    """
    Compare training curves across different methods.
    
    Args:
        histories: Dict mapping method name to training history
        metric: Metric to compare
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for method, history in histories.items():
        if metric in history:
            epochs = range(1, len(history[metric]) + 1)
            color = DRRL_COLORS.get(method.lower(), None)
            ax.plot(epochs, history[metric], label=method.upper(), color=color, linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_worst_group_comparison(
    histories: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Compare worst-group accuracy across methods.
    
    Args:
        histories: Dict mapping method name to training history
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Average accuracy
    ax = axes[0]
    for method, history in histories.items():
        if 'val_accuracy' in history:
            epochs = range(1, len(history['val_accuracy']) + 1)
            color = DRRL_COLORS.get(method.lower(), None)
            ax.plot(epochs, history['val_accuracy'], label=method.upper(), color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Average Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Worst-group accuracy
    ax = axes[1]
    for method, history in histories.items():
        if 'val_worst_group_accuracy' in history:
            epochs = range(1, len(history['val_worst_group_accuracy']) + 1)
            color = DRRL_COLORS.get(method.lower(), None)
            ax.plot(epochs, history['val_worst_group_accuracy'], 
                   label=method.upper(), color=color, linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Worst-Group Accuracy')
    ax.set_title('Worst-Group Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_learning_rate(
    lr_history: List[float],
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot learning rate schedule."""
    set_style('paper')
    fig, ax = plt.subplots(figsize=(8, 4))
    
    epochs = range(1, len(lr_history) + 1)
    ax.plot(epochs, lr_history, color=DRRL_COLORS['accent'], linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
