"""
Distribution shift visualization for the DRRL Framework.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

from viz.style import DRRL_COLORS, set_style, save_figure


def plot_id_vs_ood(
    method_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = 'ID vs OOD Performance'
) -> plt.Figure:
    """
    Plot in-distribution vs out-of-distribution accuracy.
    
    Args:
        method_results: Dict mapping method name to dict with
                       'id_accuracy' and 'ood_accuracy'
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = list(method_results.keys())
    x = np.arange(len(methods))
    width = 0.35
    
    id_accs = [method_results[m].get('id_accuracy', 0) for m in methods]
    ood_accs = [method_results[m].get('ood_accuracy', 0) for m in methods]
    
    bars1 = ax.bar(x - width/2, id_accs, width, label='In-Distribution',
                   color=DRRL_COLORS['accent'], edgecolor='black')
    bars2 = ax.bar(x + width/2, ood_accs, width, label='Out-of-Distribution',
                   color=DRRL_COLORS['secondary'], edgecolor='black')
    
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{bar.get_height():.1%}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{bar.get_height():.1%}', ha='center', fontsize=9)
    
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


def plot_ood_gap(
    method_results: Dict[str, Dict[str, float]],
    save_path: Optional[str] = None,
    title: str = 'OOD Generalization Gap'
) -> plt.Figure:
    """
    Plot OOD generalization gap across methods.
    
    Args:
        method_results: Dict with 'id_accuracy' and 'ood_accuracy'
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, ax = plt.subplots(figsize=(8, 5))
    
    methods = list(method_results.keys())
    gaps = []
    
    for m in methods:
        id_acc = method_results[m].get('id_accuracy', 0)
        ood_acc = method_results[m].get('ood_accuracy', 0)
        gaps.append(id_acc - ood_acc)
    
    colors = [DRRL_COLORS.get(m.lower(), DRRL_COLORS['primary']) for m in methods]
    
    bars = ax.bar([m.upper() for m in methods], gaps, color=colors, edgecolor='black')
    
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
               f'{gap:.1%}', ha='center', fontsize=10)
    
    ax.set_ylabel('OOD Gap (ID - OOD)')
    ax.set_title(title)
    ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_corruption_severity(
    severity_results: Dict[int, float],
    clean_accuracy: float,
    save_path: Optional[str] = None,
    title: str = 'Accuracy vs Corruption Severity'
) -> plt.Figure:
    """
    Plot accuracy degradation across corruption severity levels.
    
    Args:
        severity_results: Dict mapping severity level (1-5) to accuracy
        clean_accuracy: Accuracy on clean data
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    fig, ax = plt.subplots(figsize=(8, 6))
    
    severities = [0] + sorted(severity_results.keys())
    accuracies = [clean_accuracy] + [severity_results[s] for s in sorted(severity_results.keys())]
    
    ax.plot(severities, accuracies, 'o-', linewidth=2, markersize=8,
           color=DRRL_COLORS['primary'])
    
    ax.fill_between(severities, accuracies, alpha=0.2, color=DRRL_COLORS['primary'])
    
    ax.set_xlabel('Corruption Severity (0 = clean)')
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(severities)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_shift_comparison(
    shift_results: Dict[str, Dict[str, float]],
    shift_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 'Performance Across Distribution Shifts'
) -> plt.Figure:
    """
    Plot performance across different types of distribution shifts.
    
    Args:
        shift_results: Dict mapping shift type to dict with method accuracies
        shift_names: Display names for shifts
        save_path: Path to save figure
        
    Returns:
        Figure object
    """
    set_style('paper')
    
    shifts = list(shift_results.keys())
    methods = list(shift_results[shifts[0]].keys())
    
    if shift_names is None:
        shift_names = shifts
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(shifts))
    width = 0.8 / len(methods)
    
    for i, method in enumerate(methods):
        accs = [shift_results[s][method] for s in shifts]
        color = DRRL_COLORS.get(method.lower(), None)
        ax.bar(x + i * width, accs, width, label=method.upper(),
              color=color, edgecolor='black')
    
    ax.set_ylabel('Accuracy')
    ax.set_title(title)
    ax.set_xticks(x + width * (len(methods) - 1) / 2)
    ax.set_xticklabels(shift_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
