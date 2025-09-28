"""
Segmentation metrics using TorchMetrics.

This module provides TorchMetrics-compatible metrics for
segmentation tasks with consistent interfaces.
"""

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryJaccardIndex,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score
)
from typing import Any, Optional


class SegmentationDice(Metric):
    """
    Dice coefficient metric for binary segmentation.

    Custom implementation that works with current TorchMetrics.
    """

    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.add_state("intersection", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("union", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state."""
        # Ensure predictions are probabilities
        if preds.max() > 1.0:
            preds = torch.sigmoid(preds)

        # Apply threshold to get binary predictions
        preds_binary = (preds > self.threshold).float()
        target_binary = target.float()

        # Calculate intersection and union
        intersection = torch.sum(preds_binary * target_binary)
        union = torch.sum(preds_binary) + torch.sum(target_binary)

        self.intersection += intersection
        self.union += union

    def compute(self) -> torch.Tensor:
        """Compute Dice coefficient."""
        dice = (2.0 * self.intersection + self.smooth) / (self.union + self.smooth)
        return dice


class SegmentationIoU(Metric):
    """
    IoU metric using TorchMetrics BinaryJaccardIndex.
    """

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.jaccard = BinaryJaccardIndex(threshold=threshold)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state."""
        self.jaccard.update(preds, target.int())

    def compute(self) -> torch.Tensor:
        """Compute IoU."""
        return self.jaccard.compute()

    def reset(self) -> None:
        """Reset metric state."""
        self.jaccard.reset()


class SegmentationPrecision(Metric):
    """
    Precision metric using TorchMetrics BinaryPrecision.
    """

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.precision = BinaryPrecision(threshold=threshold)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state."""
        self.precision.update(preds, target.int())

    def compute(self) -> torch.Tensor:
        """Compute precision."""
        return self.precision.compute()

    def reset(self) -> None:
        """Reset metric state."""
        self.precision.reset()


class SegmentationRecall(Metric):
    """
    Recall metric using TorchMetrics BinaryRecall.
    """

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.recall = BinaryRecall(threshold=threshold)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state."""
        self.recall.update(preds, target.int())

    def compute(self) -> torch.Tensor:
        """Compute recall."""
        return self.recall.compute()

    def reset(self) -> None:
        """Reset metric state."""
        self.recall.reset()


class SegmentationF1(Metric):
    """
    F1 score metric using TorchMetrics BinaryF1Score.
    """

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.f1 = BinaryF1Score(threshold=threshold)

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update metric state."""
        self.f1.update(preds, target.int())

    def compute(self) -> torch.Tensor:
        """Compute F1 score."""
        return self.f1.compute()

    def reset(self) -> None:
        """Reset metric state."""
        self.f1.reset()


def get_metric(metric_name: str, threshold: float = 0.5, **kwargs) -> Metric:
    """
    Factory function to create segmentation metrics.

    Args:
        metric_name: Name of the metric
        threshold: Threshold for binary predictions
        **kwargs: Additional arguments for the metric

    Returns:
        Instantiated metric
    """
    metric_name = metric_name.lower()

    if metric_name == "dice":
        return SegmentationDice(threshold=threshold, **kwargs)
    elif metric_name == "iou":
        return SegmentationIoU(threshold=threshold, **kwargs)
    elif metric_name == "precision":
        return SegmentationPrecision(threshold=threshold, **kwargs)
    elif metric_name == "recall":
        return SegmentationRecall(threshold=threshold, **kwargs)
    elif metric_name == "f1":
        return SegmentationF1(threshold=threshold, **kwargs)
    else:
        raise ValueError(f"Unknown metric: {metric_name}")


__all__ = [
    "SegmentationDice",
    "SegmentationIoU",
    "SegmentationPrecision",
    "SegmentationRecall",
    "SegmentationF1",
    "get_metric"
]