"""
Unit tests for metrics.
"""

import pytest
import torch

from src.metrics import get_metric
from src.metrics.segmentation import (
    SegmentationDice,
    SegmentationIoU,
    SegmentationPrecision,
    SegmentationRecall,
    SegmentationF1
)


class TestSegmentationDice:
    """Test Dice metric implementation."""

    def test_dice_perfect_prediction(self):
        """Test Dice metric with perfect prediction."""
        metric = SegmentationDice(threshold=0.5)

        # Perfect prediction
        preds = torch.ones(2, 1, 16, 16)
        target = torch.ones(2, 1, 16, 16)

        metric.update(preds, target)
        dice_score = metric.compute()

        # Should be close to 1.0
        assert dice_score > 0.99

    def test_dice_worst_prediction(self):
        """Test Dice metric with worst prediction."""
        metric = SegmentationDice(threshold=0.5)

        # Worst case: no overlap
        preds = torch.ones(2, 1, 16, 16)
        target = torch.zeros(2, 1, 16, 16)

        metric.update(preds, target)
        dice_score = metric.compute()

        # Should be close to 0.0
        assert dice_score < 0.1

    def test_dice_partial_overlap(self):
        """Test Dice metric with partial overlap."""
        metric = SegmentationDice(threshold=0.5)

        # Half overlap
        preds = torch.zeros(1, 1, 4, 4)
        preds[0, 0, :2, :] = 1.0  # First half is 1

        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 1:3, :] = 1.0  # Middle half is 1

        metric.update(preds, target)
        dice_score = metric.compute()

        # Should be between 0 and 1
        assert 0 < dice_score < 1

    def test_dice_reset(self):
        """Test that metric reset works correctly."""
        metric = SegmentationDice(threshold=0.5)

        # First computation
        preds = torch.ones(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        metric.update(preds, target)
        score1 = metric.compute()

        # Reset and compute again with different data
        metric.reset()
        preds = torch.zeros(1, 1, 4, 4)
        target = torch.ones(1, 1, 4, 4)
        metric.update(preds, target)
        score2 = metric.compute()

        # Scores should be different
        assert abs(score1 - score2) > 0.5

    def test_dice_threshold_effect(self):
        """Test that threshold affects Dice computation."""
        metric_low = SegmentationDice(threshold=0.3)
        metric_high = SegmentationDice(threshold=0.7)

        # Probabilities around 0.5
        preds = torch.ones(1, 1, 4, 4) * 0.5
        target = torch.ones(1, 1, 4, 4)

        metric_low.update(preds.clone(), target.clone())
        metric_high.update(preds.clone(), target.clone())

        score_low = metric_low.compute()
        score_high = metric_high.compute()

        # Low threshold should give higher Dice (more predictions above threshold)
        assert score_low > score_high


class TestMetricFactory:
    """Test metric factory function."""

    def test_get_dice_metric(self):
        """Test getting Dice metric from factory."""
        metric = get_metric("dice", threshold=0.6)

        assert isinstance(metric, SegmentationDice)
        assert metric.threshold == 0.6

    def test_get_iou_metric(self):
        """Test getting IoU metric from factory."""
        metric = get_metric("iou", threshold=0.4)

        assert isinstance(metric, SegmentationIoU)
        assert metric.threshold == 0.4

    def test_get_precision_metric(self):
        """Test getting Precision metric from factory."""
        metric = get_metric("precision")

        assert isinstance(metric, SegmentationPrecision)

    def test_get_recall_metric(self):
        """Test getting Recall metric from factory."""
        metric = get_metric("recall")

        assert isinstance(metric, SegmentationRecall)

    def test_get_f1_metric(self):
        """Test getting F1 metric from factory."""
        metric = get_metric("f1")

        assert isinstance(metric, SegmentationF1)

    def test_unknown_metric_raises_error(self):
        """Test that unknown metric raises error."""
        with pytest.raises(ValueError, match="Unknown metric"):
            get_metric("unknown_metric")


class TestMetricCompatibility:
    """Test that metrics work with TorchMetrics interface."""

    def test_all_metrics_have_required_methods(self):
        """Test that all metrics have required TorchMetrics methods."""
        metric_names = ["dice", "iou", "precision", "recall", "f1"]

        for name in metric_names:
            metric = get_metric(name)

            # Check required methods exist
            assert hasattr(metric, "update")
            assert hasattr(metric, "compute")
            assert hasattr(metric, "reset")

            # Check they are callable
            assert callable(metric.update)
            assert callable(metric.compute)
            assert callable(metric.reset)

    def test_metrics_work_with_different_shapes(self):
        """Test that metrics handle different input shapes."""
        metric = get_metric("dice")

        # Test with 4D tensors [B, C, H, W]
        preds_4d = torch.rand(2, 1, 8, 8)
        target_4d = torch.randint(0, 2, (2, 1, 8, 8)).float()

        metric.update(preds_4d, target_4d)
        score_4d = metric.compute()
        assert isinstance(score_4d, torch.Tensor)

        # Reset and test with 3D tensors [B, H, W]
        metric.reset()
        preds_3d = torch.rand(2, 8, 8)
        target_3d = torch.randint(0, 2, (2, 8, 8)).float()

        metric.update(preds_3d, target_3d)
        score_3d = metric.compute()
        assert isinstance(score_3d, torch.Tensor)

    def test_metrics_accumulate_correctly(self):
        """Test that metrics accumulate over multiple updates."""
        metric = SegmentationDice(threshold=0.5)

        # First batch - perfect prediction
        preds1 = torch.ones(1, 1, 4, 4)
        target1 = torch.ones(1, 1, 4, 4)
        metric.update(preds1, target1)

        # Second batch - worst prediction
        preds2 = torch.ones(1, 1, 4, 4)
        target2 = torch.zeros(1, 1, 4, 4)
        metric.update(preds2, target2)

        # Combined score should be between 0 and 1
        combined_score = metric.compute()
        assert 0 < combined_score < 1

        # Should be around 0.5 (average of 1.0 and 0.0)
        assert 0.3 < combined_score < 0.7