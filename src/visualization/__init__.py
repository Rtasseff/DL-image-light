"""
Visualization module for segmentation platform.

This module provides utilities for visualizing training results,
prediction overlays, and evaluation metrics.
"""

from .overlays import (
    apply_colormap,
    create_overlay,
    create_comparison_grid,
    generate_overlays,
    create_error_visualization
)

from .metrics_plot import (
    plot_training_curves,
    plot_loss_curves,
    plot_metrics_summary,
    create_confusion_matrix_plot,
    plot_learning_rate_schedule,
    generate_all_plots
)

__all__ = [
    # Overlay functions
    "apply_colormap",
    "create_overlay",
    "create_comparison_grid",
    "generate_overlays",
    "create_error_visualization",

    # Plotting functions
    "plot_training_curves",
    "plot_loss_curves",
    "plot_metrics_summary",
    "create_confusion_matrix_plot",
    "plot_learning_rate_schedule",
    "generate_all_plots"
]