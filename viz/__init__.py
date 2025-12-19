"""
Visualization module for the DRRL Framework.
"""

from viz.style import (
    set_style,
    get_figure,
    save_figure,
    DRRL_COLORS,
    GROUP_COLORS,
    CLASS_COLORS
)
from viz.training_curves import (
    plot_training_curves,
    plot_method_comparison,
    plot_worst_group_comparison,
    plot_learning_rate
)
from viz.embeddings import (
    plot_embeddings_umap,
    plot_embeddings_tsne,
    plot_embeddings_by_group_and_class
)
from viz.group_performance import (
    plot_group_accuracies,
    plot_worst_vs_average,
    plot_robustness_gap,
    plot_group_distribution
)
from viz.distribution_shift import (
    plot_id_vs_ood,
    plot_ood_gap,
    plot_corruption_severity,
    plot_shift_comparison
)
from viz.confusion_plots import (
    plot_confusion_matrix,
    plot_per_group_confusion,
    plot_error_analysis
)


__all__ = [
    'set_style',
    'save_figure',
    'DRRL_COLORS',
    'plot_training_curves',
    'plot_method_comparison',
    'plot_worst_group_comparison',
    'plot_embeddings_umap',
    'plot_embeddings_tsne',
    'plot_embeddings_by_group_and_class',
    'plot_group_accuracies',
    'plot_worst_vs_average',
    'plot_robustness_gap',
    'plot_id_vs_ood',
    'plot_ood_gap',
    'plot_confusion_matrix',
    'plot_per_group_confusion',
    'plot_error_analysis',
]
