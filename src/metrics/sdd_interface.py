"""
SDD v4.1 compliant metrics interface.

This module provides a wrapper that ensures metrics conform to the
SDD Metrics contract: compute() -> Dict[str, Tensor]
"""

from typing import Dict
import torch
from torch import Tensor

from .segmentation import (
    SegmentationDice,
    SegmentationIoU,
    SegmentationPrecision,
    SegmentationRecall,
    SegmentationF1
)


class SDDMetricsWrapper:
    """
    SDD v4.1 compliant metrics wrapper.

    This class implements the stable Metrics interface contract:
    - update(pred: Tensor, target: Tensor) -> None
    - compute() -> Dict[str, Tensor]
    - reset() -> None
    """

    def __init__(self, metric_names: list = None):
        """
        Initialize metrics wrapper.

        Args:
            metric_names: List of metric names to include
                         (default: ["dice", "iou"])
        """
        if metric_names is None:
            metric_names = ["dice", "iou"]

        self.metrics = {}
        for name in metric_names:
            if name == "dice":
                self.metrics[name] = SegmentationDice()
            elif name == "iou":
                self.metrics[name] = SegmentationIoU()
            elif name == "precision":
                self.metrics[name] = SegmentationPrecision()
            elif name == "recall":
                self.metrics[name] = SegmentationRecall()
            elif name == "f1":
                self.metrics[name] = SegmentationF1()
            else:
                raise ValueError(f"Unknown metric: {name}")

    def update(self, pred: Tensor, target: Tensor) -> None:
        """
        Update all metrics.

        Args:
            pred: Predictions [N, H, W] or [N, 1, H, W]
            target: Ground truth [N, H, W] or [N, 1, H, W]
        """
        for metric in self.metrics.values():
            metric.update(pred, target)

    def compute(self) -> Dict[str, Tensor]:
        """
        Compute all metrics.

        Returns:
            Dictionary mapping metric names to computed values
        """
        results = {}
        for name, metric in self.metrics.items():
            results[name] = metric.compute()
        return results

    def reset(self) -> None:
        """Reset all metrics."""
        for metric in self.metrics.values():
            metric.reset()


def create_sdd_metrics(metric_names: list = None) -> SDDMetricsWrapper:
    """
    Factory function to create SDD-compliant metrics.

    Args:
        metric_names: List of metric names

    Returns:
        SDDMetricsWrapper instance
    """
    return SDDMetricsWrapper(metric_names)


__all__ = ["SDDMetricsWrapper", "create_sdd_metrics"]