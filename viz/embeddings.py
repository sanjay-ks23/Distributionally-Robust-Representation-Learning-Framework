"""
Embedding visualization using UMAP and t-SNE.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Union, List
from pathlib import Path

from viz.style import GROUP_COLORS, CLASS_COLORS, set_style, save_figure


def plot_embeddings_umap(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 'UMAP Embedding Visualization',
    n_neighbors: int = 15,
    min_dist: float = 0.1
) -> plt.Figure:
    """
    Visualize embeddings using UMAP.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: Array of labels for coloring
        label_names: Names for each label
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Figure object
    """
    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn: pip install umap-learn")
    
    set_style('paper')
    
    # Reduce to 2D
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    unique_labels = np.unique(labels)
    colors = CLASS_COLORS if len(unique_labels) <= 2 else GROUP_COLORS
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names else f'Class {label}'
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=colors[i % len(colors)],
            label=name,
            alpha=0.6,
            s=30
        )
    
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_embeddings_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    title: str = 't-SNE Embedding Visualization',
    perplexity: int = 30
) -> plt.Figure:
    """
    Visualize embeddings using t-SNE.
    
    Args:
        embeddings: Array of shape (n_samples, embedding_dim)
        labels: Array of labels for coloring
        label_names: Names for each label
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Figure object
    """
    from sklearn.manifold import TSNE
    
    set_style('paper')
    
    # Reduce to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    embedding_2d = tsne.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    unique_labels = np.unique(labels)
    colors = CLASS_COLORS if len(unique_labels) <= 2 else GROUP_COLORS
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        name = label_names[label] if label_names else f'Class {label}'
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=colors[i % len(colors)],
            label=name,
            alpha=0.6,
            s=30
        )
    
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig


def plot_embeddings_by_group_and_class(
    embeddings: np.ndarray,
    targets: np.ndarray,
    groups: np.ndarray,
    class_names: Optional[List[str]] = None,
    group_names: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    method: str = 'umap'
) -> plt.Figure:
    """
    Create side-by-side embedding plots colored by class and by group.
    
    Args:
        embeddings: Embedding array
        targets: Class labels
        groups: Group labels
        class_names: Names for classes
        group_names: Names for groups
        save_path: Path to save figure
        method: 'umap' or 'tsne'
        
    Returns:
        Figure object
    """
    set_style('paper')
    
    # Reduce dimensionality
    if method == 'umap':
        try:
            import umap
            reducer = umap.UMAP(random_state=42)
        except ImportError:
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
    
    embedding_2d = reducer.fit_transform(embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Color by class
    ax = axes[0]
    for c in np.unique(targets):
        mask = targets == c
        name = class_names[c] if class_names else f'Class {c}'
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=CLASS_COLORS[c % len(CLASS_COLORS)],
            label=name,
            alpha=0.6,
            s=30
        )
    ax.set_title('Colored by Class')
    ax.legend()
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    # Color by group
    ax = axes[1]
    for g in np.unique(groups):
        mask = groups == g
        name = group_names[g] if group_names else f'Group {g}'
        ax.scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=GROUP_COLORS[g % len(GROUP_COLORS)],
            label=name,
            alpha=0.6,
            s=30
        )
    ax.set_title('Colored by Group (Spurious Feature)')
    ax.legend()
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')
    
    plt.tight_layout()
    
    if save_path:
        save_figure(fig, save_path)
    
    return fig
