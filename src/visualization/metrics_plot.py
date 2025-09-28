"""
Training metrics visualization utilities.

This module provides functions to plot training curves, confusion matrices,
and other metrics visualizations for model evaluation.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from ..utils.logging import get_logger

logger = get_logger(__name__)


def plot_training_curves(
    metrics_csv: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (15, 10)
) -> None:
    """
    Plot training and validation curves from Lightning CSV logs.

    Args:
        metrics_csv: Path to Lightning metrics CSV file
        output_path: Path to save the plot
        figsize: Figure size (width, height)
    """
    # Read metrics CSV
    df = pd.read_csv(metrics_csv)

    # Determine available metrics
    train_metrics = [col for col in df.columns if col.startswith('train/') and not col.endswith('_step')]
    val_metrics = [col for col in df.columns if col.startswith('val/')]

    # Create subplots
    n_metrics = len(set([m.split('/')[-1] for m in train_metrics + val_metrics]))
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    # Plot each metric
    metric_names = set()
    for col in train_metrics + val_metrics:
        metric_name = col.split('/')[-1]
        metric_names.add(metric_name)

    for idx, metric_name in enumerate(sorted(metric_names)):
        if idx >= len(axes):
            break

        ax = axes[idx]

        # Plot training curve
        train_col = f'train/{metric_name}'
        if train_col in df.columns:
            train_data = df[train_col].dropna()
            epochs = df.loc[train_data.index, 'epoch'].values
            ax.plot(epochs, train_data.values, label=f'Train {metric_name}', marker='o', alpha=0.7)

        # Plot validation curve
        val_col = f'val/{metric_name}'
        if val_col in df.columns:
            val_data = df[val_col].dropna()
            epochs = df.loc[val_data.index, 'epoch'].values
            ax.plot(epochs, val_data.values, label=f'Val {metric_name}', marker='s', alpha=0.7)

        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f'{metric_name.capitalize()} vs Epoch')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Remove extra subplots
    for idx in range(len(metric_names), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training curves saved to {output_path}")
    else:
        plt.show()

    plt.close(fig)


def plot_loss_curves(
    metrics_csv: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot training and validation loss curves.

    Args:
        metrics_csv: Path to Lightning metrics CSV file
        output_path: Path to save the plot
        figsize: Figure size
    """
    df = pd.read_csv(metrics_csv)

    plt.figure(figsize=figsize)

    # Plot training loss
    if 'train/loss_epoch' in df.columns:
        train_loss = df['train/loss_epoch'].dropna()
        epochs = df.loc[train_loss.index, 'epoch'].values
        plt.plot(epochs, train_loss.values, label='Training Loss', marker='o', alpha=0.7)

    # Plot validation loss
    if 'val/loss' in df.columns:
        val_loss = df['val/loss'].dropna()
        epochs = df.loc[val_loss.index, 'epoch'].values
        plt.plot(epochs, val_loss.values, label='Validation Loss', marker='s', alpha=0.7)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Loss curves saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_metrics_summary(
    metrics_dict: Dict[str, float],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot summary bar chart of final metrics.

    Args:
        metrics_dict: Dictionary of metric names and values
        output_path: Path to save the plot
        figsize: Figure size
    """
    # Filter out loss metrics for cleaner visualization
    display_metrics = {k: v for k, v in metrics_dict.items()
                      if not k.endswith('loss') and not k.endswith('loss_epoch')}

    if not display_metrics:
        display_metrics = metrics_dict

    plt.figure(figsize=figsize)

    metrics_names = list(display_metrics.keys())
    metrics_values = list(display_metrics.values())

    # Create bar plot
    bars = plt.bar(metrics_names, metrics_values, alpha=0.7, color='skyblue', edgecolor='navy')

    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.ylabel('Score')
    plt.title('Final Metrics Summary')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.1)
    plt.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metrics summary saved to {output_path}")
    else:
        plt.show()

    plt.close()


def create_confusion_matrix_plot(
    true_positives: int,
    false_positives: int,
    false_negatives: int,
    true_negatives: int,
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 6)
) -> None:
    """
    Create confusion matrix visualization for binary segmentation.

    Args:
        true_positives: Number of true positive pixels
        false_positives: Number of false positive pixels
        false_negatives: Number of false negative pixels
        true_negatives: Number of true negative pixels
        output_path: Path to save the plot
        figsize: Figure size
    """
    # Create confusion matrix
    cm = np.array([[true_negatives, false_positives],
                   [false_negatives, true_positives]])

    plt.figure(figsize=figsize)

    # Plot heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'],
                cbar_kws={'label': 'Number of Pixels'})

    plt.title('Confusion Matrix (Pixel-wise)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_learning_rate_schedule(
    metrics_csv: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Plot learning rate schedule over training.

    Args:
        metrics_csv: Path to Lightning metrics CSV file
        output_path: Path to save the plot
        figsize: Figure size
    """
    df = pd.read_csv(metrics_csv)

    if 'lr-SGD' in df.columns:
        lr_col = 'lr-SGD'
    elif 'lr-Adam' in df.columns:
        lr_col = 'lr-Adam'
    elif 'lr-AdamW' in df.columns:
        lr_col = 'lr-AdamW'
    else:
        logger.warning("No learning rate column found in metrics CSV")
        return

    plt.figure(figsize=figsize)

    lr_data = df[lr_col].dropna()
    epochs = df.loc[lr_data.index, 'epoch'].values

    plt.plot(epochs, lr_data.values, marker='o', alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning rate schedule saved to {output_path}")
    else:
        plt.show()

    plt.close()


def generate_all_plots(
    run_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Generate all available plots for a training run.

    Args:
        run_dir: Directory containing training outputs
        output_dir: Directory to save plots (defaults to run_dir/plots)
    """
    run_dir = Path(run_dir)

    if output_dir is None:
        output_dir = run_dir / "plots"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating plots for run: {run_dir}")

    # Find metrics CSV
    metrics_csv = None
    for csv_file in run_dir.rglob("metrics.csv"):
        metrics_csv = csv_file
        break

    if metrics_csv and metrics_csv.exists():
        logger.info(f"Found metrics CSV: {metrics_csv}")

        # Plot training curves
        plot_training_curves(
            metrics_csv,
            output_dir / "training_curves.png"
        )

        # Plot loss curves
        plot_loss_curves(
            metrics_csv,
            output_dir / "loss_curves.png"
        )

        # Plot learning rate schedule
        plot_learning_rate_schedule(
            metrics_csv,
            output_dir / "lr_schedule.png"
        )

    # Load final metrics if available
    final_metrics_path = run_dir / "final_metrics.json"
    if final_metrics_path.exists():
        with open(final_metrics_path, 'r') as f:
            final_metrics = json.load(f)

        plot_metrics_summary(
            final_metrics,
            output_dir / "metrics_summary.png"
        )

    logger.info(f"Plots saved to {output_dir}")


__all__ = [
    "plot_training_curves",
    "plot_loss_curves",
    "plot_metrics_summary",
    "create_confusion_matrix_plot",
    "plot_learning_rate_schedule",
    "generate_all_plots"
]