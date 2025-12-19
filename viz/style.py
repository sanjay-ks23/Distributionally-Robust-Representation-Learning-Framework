"""
Plotting style configuration for the DRRL Framework.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple


# Color palettes
DRRL_COLORS = {
    'erm': '#1f77b4',    # Blue
    'sam': '#ff7f0e',    # Orange
    'dro': '#2ca02c',    # Green
    'primary': '#2C3E50',
    'secondary': '#E74C3C',
    'accent': '#3498DB',
}

GROUP_COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
CLASS_COLORS = ['#1f77b4', '#ff7f0e']


def set_style(style: str = 'paper'):
    """Set plotting style for publication-quality figures."""
    if style == 'paper':
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.figsize': (8, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })
    elif style == 'presentation':
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams.update({
            'font.size': 16,
            'axes.titlesize': 20,
            'axes.labelsize': 16,
            'figure.figsize': (12, 8),
        })
    else:
        plt.style.use('default')


def get_figure(
    nrows: int = 1,
    ncols: int = 1,
    figsize: Optional[Tuple[float, float]] = None
) -> Tuple[plt.Figure, plt.Axes]:
    """Create a figure with consistent styling."""
    if figsize is None:
        figsize = (6 * ncols, 5 * nrows)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def save_figure(
    fig: plt.Figure,
    path: str,
    formats: list = ['png', 'pdf']
):
    """Save figure in multiple formats."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        fig.savefig(
            path.with_suffix(f'.{fmt}'),
            dpi=300,
            bbox_inches='tight'
        )


def add_legend(
    ax: plt.Axes,
    loc: str = 'upper right',
    frameon: bool = True
):
    """Add styled legend to axes."""
    ax.legend(
        loc=loc,
        frameon=frameon,
        fancybox=True,
        framealpha=0.8
    )


def format_axis(
    ax: plt.Axes,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    xlim: Optional[Tuple] = None,
    ylim: Optional[Tuple] = None
):
    """Format axis with labels and limits."""
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)
