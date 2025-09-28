"""
Unit tests for loss functions.
"""

import pytest
import torch
import torch.nn as nn

from src.losses import get_loss
from src.losses.dice import DiceLoss, TverskyLoss, CompoundLoss


class TestDiceLoss:
    """Test Dice loss implementation."""

    def test_dice_loss_computation(self):
        """Test basic Dice loss computation."""
        loss_fn = DiceLoss(smooth=1.0)

        # Create simple test data
        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()

        loss = loss_fn(pred, target)

        # Loss should be a scalar between 0 and 1
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert 0 <= loss <= 1

    def test_dice_loss_perfect_prediction(self):
        """Test Dice loss with perfect prediction."""
        loss_fn = DiceLoss(smooth=1.0)

        # Perfect prediction (target == pred)
        target = torch.ones(2, 1, 16, 16)
        pred = torch.ones(2, 1, 16, 16)

        loss = loss_fn(pred, target)

        # Loss should be very close to 0
        assert loss < 0.1

    def test_dice_loss_worst_prediction(self):
        """Test Dice loss with worst case prediction."""
        loss_fn = DiceLoss(smooth=1.0)

        # Worst case: target is 1, pred is 0
        target = torch.ones(2, 1, 16, 16)
        pred = torch.zeros(2, 1, 16, 16)

        loss = loss_fn(pred, target)

        # Loss should be close to 1
        assert loss > 0.9

    def test_dice_loss_with_logits(self):
        """Test that Dice loss handles logits correctly."""
        loss_fn = DiceLoss(smooth=1.0)

        # Test with logits (should apply sigmoid internally)
        logits = torch.randn(2, 1, 16, 16) * 5  # Large range
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()

        loss = loss_fn(logits, target)

        assert isinstance(loss, torch.Tensor)
        assert 0 <= loss <= 1

    def test_dice_loss_different_shapes(self):
        """Test Dice loss with different input shapes."""
        loss_fn = DiceLoss(smooth=1.0)

        # Test 3D input (should expand to 4D)
        pred_3d = torch.sigmoid(torch.randn(2, 32, 32))
        target_3d = torch.randint(0, 2, (2, 32, 32)).float()

        loss = loss_fn(pred_3d, target_3d)
        assert isinstance(loss, torch.Tensor)


class TestTverskyLoss:
    """Test Tversky loss implementation."""

    def test_tversky_loss_computation(self):
        """Test basic Tversky loss computation."""
        loss_fn = TverskyLoss(alpha=0.5, beta=0.5, smooth=1.0)

        pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
        target = torch.randint(0, 2, (2, 1, 32, 32)).float()

        loss = loss_fn(pred, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert 0 <= loss <= 1

    def test_tversky_reduces_to_dice(self):
        """Test that Tversky with alpha=beta=0.5 is similar to Dice."""
        dice_loss = DiceLoss(smooth=1.0)
        tversky_loss = TverskyLoss(alpha=0.5, beta=0.5, smooth=1.0)

        # Use same input for both
        torch.manual_seed(42)
        pred = torch.sigmoid(torch.randn(2, 1, 16, 16))
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()

        dice_val = dice_loss(pred, target)
        tversky_val = tversky_loss(pred, target)

        # Should be approximately equal
        assert abs(dice_val - tversky_val) < 0.1

    def test_tversky_alpha_beta_effects(self):
        """Test that different alpha/beta values affect loss differently."""
        # High alpha (penalize false positives more)
        loss_high_alpha = TverskyLoss(alpha=0.7, beta=0.3)

        # High beta (penalize false negatives more)
        loss_high_beta = TverskyLoss(alpha=0.3, beta=0.7)

        # Create scenario with false positives
        pred = torch.ones(1, 1, 16, 16) * 0.8  # High predictions
        target = torch.zeros(1, 1, 16, 16)     # All zeros (many false positives)

        loss_alpha = loss_high_alpha(pred, target)
        loss_beta = loss_high_beta(pred, target)

        # High alpha should give higher loss for false positives
        assert loss_alpha > loss_beta


class TestLossFactory:
    """Test loss factory function."""

    def test_get_dice_loss(self):
        """Test getting Dice loss from factory."""
        loss_fn = get_loss("dice", {"smooth": 2.0})

        assert isinstance(loss_fn, DiceLoss)
        assert loss_fn.smooth == 2.0

    def test_get_tversky_loss(self):
        """Test getting Tversky loss from factory."""
        loss_fn = get_loss("tversky", {"alpha": 0.3, "beta": 0.7})

        assert isinstance(loss_fn, TverskyLoss)
        assert loss_fn.alpha == 0.3
        assert loss_fn.beta == 0.7

    def test_get_bce_loss(self):
        """Test getting BCE loss from factory."""
        loss_fn = get_loss("bce")

        assert isinstance(loss_fn, nn.BCEWithLogitsLoss)

    def test_unknown_loss_raises_error(self):
        """Test that unknown loss type raises error."""
        with pytest.raises(ValueError, match="Unknown loss type"):
            get_loss("unknown_loss")

    def test_focal_loss_not_implemented(self):
        """Test that focal loss raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            get_loss("focal")


class TestCompoundLoss:
    """Test compound loss implementation."""

    def test_compound_loss_computation(self):
        """Test basic compound loss computation."""
        losses = {
            "dice": (DiceLoss(), 1.0),
            "bce": (nn.BCEWithLogitsLoss(), 0.5)
        }
        compound_loss = CompoundLoss(losses)

        logits = torch.randn(2, 1, 16, 16)
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()

        loss = compound_loss(logits, target)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0

    def test_compound_loss_weighting(self):
        """Test that compound loss weighting works correctly."""
        # Create two identical losses with different weights
        losses_weight_1 = {
            "dice1": (DiceLoss(), 1.0),
            "dice2": (DiceLoss(), 1.0)
        }
        losses_weight_2 = {
            "dice1": (DiceLoss(), 1.0),
            "dice2": (DiceLoss(), 2.0)  # Double weight
        }

        compound_1 = CompoundLoss(losses_weight_1)
        compound_2 = CompoundLoss(losses_weight_2)

        torch.manual_seed(42)
        pred = torch.sigmoid(torch.randn(2, 1, 16, 16))
        target = torch.randint(0, 2, (2, 1, 16, 16)).float()

        loss_1 = compound_1(pred, target)
        loss_2 = compound_2(pred, target)

        # Loss_2 should be higher due to increased weight
        assert loss_2 > loss_1