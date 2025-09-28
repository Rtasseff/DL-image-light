"""
Metrics for segmentation tasks.

This module provides TorchMetrics-compatible metrics for
evaluating segmentation model performance.
"""

from .segmentation import (
    SegmentationDice,
    SegmentationIoU,
    SegmentationPrecision,
    SegmentationRecall,
    SegmentationF1,
    get_metric
)

__all__ = [
    "SegmentationDice",
    "SegmentationIoU",
    "SegmentationPrecision",
    "SegmentationRecall",
    "SegmentationF1",
    "get_metric"
]