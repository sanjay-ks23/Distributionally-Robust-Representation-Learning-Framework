"""
Group performance visualization for the DRRL Framework.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from viz.style import GROUP_COLORS, DRRL_COLORS, set_style, save_figure


def plot_group_accuracies(
    group_accuracies: Dict[int, float],
    group_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 'Per-Group Accuracy'
) -> plt.Figure:
    """
    Plot bar chart of per-group accuracies.
    
    Args:
        group_accuracies: Dict mapping group index to accuracy
        group_names: Names for each group
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    groups = sorted(group_accuracies.keys())
    accuracies = [group_accuracies[g] for g in groups]
    
    if group_names is None:
        group_names = [f'Group {g}' for g in groups]
    
    colors = [GROUP_COLORS[g % len(GROUP_COLORS)] for g in groups]
    
    bars = ax.bar(group_names, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f'{acc:.1%}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    # Add average line
    avg = np.mean(accuracies)
    ax.axhline(y=avg, color='black', linestyle='--', linewidth=1.5, label=f'Average: {avg:.1%}')
    
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_ylim(0, 1.1)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_worst_vs_average(
    method_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = 'Worst-Group vs Average Accuracy'
) -> plt.Figure:
    """
    Plot worst-group vs average accuracy comparison across methods.
    
    Args:
        method_results: Dict mapping method name to dict with 
                       'accuracy' and 'worst_group_accuracy'
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(method_results.keys())
    x = np.arange(len(methods))
    width = 0.35
    
    avg_accs = [method_results[m].get('accuracy', 0) for m in methods]
    worst_accs = [method_results[m].get('worst_group_accuracy', 0) for m in methods]
    
    bars1 = ax.bar(x - width/2, avg_accs, width, label='Average', 
                   color=DRRL_COLORS['accent'], edgecolor='black')
    bars2 = ax.bar(x + width/2, worst_accs, width, label='Worst-Group',
                   color=DRRL_COLORS['secondary'], edgecolor='black')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{height:.1%}',
                ha='center',
                va='bottom',
                fontsize=9
            )
    
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in methods])
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_robustness_gap(
    method_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = 'Robustness Gap (Average - Worst Group)'
) -> plt.Figure:
    """
    Plot robustness gap (difference between average and worst-group accuracy).
    
    Args:
        method_results: Dict mapping method name to metrics dict
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = list(method_results.keys())
    gaps = []
    
    for m in methods:
        avg = method_results[m].get('accuracy', 0)
        worst = method_results[m].get('worst_group_accuracy', 0)
        gaps.append(avg - worst)
    
    colors = [DRRL_COLORS.get(m.lower(), DRRL_COLORS['primary']) for m in methods]
    
    bars = ax.bar([m.upper() for m in methods], gaps, color=colors, edgecolor='black')
    
    # Add value labels
    for bar, gap in zip(bars, gaps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f'{gap:.1%}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    ax.set_ylabel('Robustness Gap')
    ax.set_title(title)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_group_distribution(
    group_counts: np.ndarray,
    group_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 'Group Distribution'
) -> plt.Figure:
    """
    Plot pie chart of group distribution.
    
    Args:
        group_counts: Array of counts per group
        group_names: Names for each group
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, ax = plt.subplots(figsize=(8, 8))
    
    if group_names is None:
        group_names = [f'Group {i}' for i in range(len(group_counts))]
    
    colors = GROUP_COLORS[:len(group_counts)]
    
    ax.pie(
        group_counts,
        labels=group_names,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        explode=[0.02] * len(group_counts)
    )
    
    ax.set_title(title)
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
