"""
Metrics for segmentation tasks.

This module provides metrics for evaluating segmentation model performance.
Falls back to simple implementations if torchmetrics is not available.
"""

try:
    # Try to import full TorchMetrics implementation
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

except ImportError:
    # Fallback to simple implementations for testing
    from .simple_metrics import (
        compute_iou,
        compute_precision,
        compute_recall,
        compute_dice
    )

    # Create simple get_metric function
    def get_metric(metric_type: str):
        """Simple metric factory for testing."""
        metric_map = {
            'iou': compute_iou,
            'precision': compute_precision,
            'recall': compute_recall,
            'dice': compute_dice
        }
        if metric_type not in metric_map:
            raise ValueError(f"Unknown metric: {metric_type}")
        return metric_map[metric_type]

    __all__ = [
        "compute_iou",
        "compute_precision",
        "compute_recall",
        "compute_dice",
        "get_metric"
    ]