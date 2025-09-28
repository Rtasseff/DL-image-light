"""
Dice loss implementations for segmentation tasks.

This module provides various Dice loss functions commonly used
in medical image segmentation, including soft and hard Dice variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class DiceLoss(nn.Module):
    """
    Soft Dice loss for segmentation tasks.

    The Dice loss is based on the Dice coefficient (F1 score) and is
    particularly effective for segmentation tasks with class imbalance.

    Args:
        smooth: Smoothing factor to avoid division by zero
        ignore_index: Class index to ignore in loss computation
        reduction: Reduction method ('mean', 'sum', 'none')

    Example:
        >>> loss_fn = DiceLoss(smooth=1.0)
        >>> pred = torch.sigmoid(torch.randn(2, 1, 256, 256))
        >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: Optional[int] = None,
        reduction: str = "mean"
    ):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.

        Args:
            pred: Predicted probabilities [B, C, H, W] or [B, H, W]
            target: Ground truth labels [B, C, H, W] or [B, H, W]

        Returns:
            Dice loss value
        """
        # Ensure pred is between 0 and 1 (apply sigmoid if needed)
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Handle different input shapes
        if pred.dim() == 3:  # [B, H, W]
            pred = pred.unsqueeze(1)  # [B, 1, H, W]
        if target.dim() == 3:  # [B, H, W]
            target = target.unsqueeze(1)  # [B, 1, H, W]

        # Flatten tensors
        pred_flat = pred.view(pred.size(0), pred.size(1), -1)  # [B, C, N]
        target_flat = target.view(target.size(0), target.size(1), -1)  # [B, C, N]

        # Handle ignore_index
        if self.ignore_index is not None:
            mask = target_flat != self.ignore_index
            pred_flat = pred_flat * mask.float()
            target_flat = target_flat * mask.float()

        # Compute Dice coefficient
        intersection = (pred_flat * target_flat).sum(dim=2)  # [B, C]
        union = pred_flat.sum(dim=2) + target_flat.sum(dim=2)  # [B, C]

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice

        # Apply reduction
        if self.reduction == "mean":
            return dice_loss.mean()
        elif self.reduction == "sum":
            return dice_loss.sum()
        else:
            return dice_loss


class TverskyLoss(nn.Module):
    """
    Tversky loss for segmentation with adjustable precision/recall trade-off.

    The Tversky loss is a generalization of Dice loss that allows controlling
    the trade-off between precision and recall through alpha and beta parameters.

    Args:
        alpha: Weight for false positives (controls precision)
        beta: Weight for false negatives (controls recall)
        smooth: Smoothing factor to avoid division by zero

    Example:
        >>> # Emphasize recall (good for detecting small objects)
        >>> loss_fn = TverskyLoss(alpha=0.3, beta=0.7)
        >>> pred = torch.sigmoid(torch.randn(2, 1, 256, 256))
        >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        >>> loss = loss_fn(pred, target)
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Tversky loss.

        Args:
            pred: Predicted probabilities [B, C, H, W] or [B, H, W]
            target: Ground truth labels [B, C, H, W] or [B, H, W]

        Returns:
            Tversky loss value
        """
        # Ensure pred is between 0 and 1
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        # Handle different input shapes
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Flatten tensors
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1)

        # Compute Tversky components
        true_pos = (pred_flat * target_flat).sum(dim=1)
        false_neg = (target_flat * (1 - pred_flat)).sum(dim=1)
        false_pos = ((1 - target_flat) * pred_flat).sum(dim=1)

        tversky = (true_pos + self.smooth) / (
            true_pos + self.alpha * false_pos + self.beta * false_neg + self.smooth
        )

        return (1 - tversky).mean()


class CompoundLoss(nn.Module):
    """
    Compound loss combining multiple loss functions.

    Allows combining different loss functions with specified weights,
    useful for leveraging benefits of multiple loss types.

    Args:
        losses: Dictionary mapping loss names to (loss_fn, weight) tuples

    Example:
        >>> losses = {
        ...     "dice": (DiceLoss(), 1.0),
        ...     "bce": (nn.BCEWithLogitsLoss(), 0.5)
        ... }
        >>> loss_fn = CompoundLoss(losses)
    """

    def __init__(self, losses: dict):
        super().__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}

        for name, (loss_fn, weight) in losses.items():
            self.losses[name] = loss_fn
            self.weights[name] = weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute compound loss.

        Args:
            pred: Predicted values
            target: Ground truth values

        Returns:
            Weighted sum of all losses
        """
        total_loss = 0.0

        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(pred, target)
            weighted_loss = self.weights[name] * loss_value
            total_loss += weighted_loss

        return total_loss