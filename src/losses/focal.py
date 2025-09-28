"""
Focal loss implementations for segmentation tasks.

The Focal loss addresses class imbalance by down-weighting well-classified
examples and focusing learning on hard negatives.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance in segmentation.

    The Focal Loss is designed to address class imbalance by down-weighting
    easy examples and focusing on hard examples. It modifies the standard
    cross-entropy loss by adding a focusing parameter gamma.

    Args:
        alpha: Weighting factor for rare class (typically 0.25)
        gamma: Focusing parameter (typically 2.0)
        reduction: Reduction method ('mean', 'sum', 'none')

    Example:
        >>> loss_fn = FocalLoss(alpha=0.25, gamma=2.0)
        >>> logits = torch.randn(2, 1, 256, 256)
        >>> target = torch.randint(0, 2, (2, 1, 256, 256)).float()
        >>> loss = loss_fn(logits, target)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal loss.

        Args:
            logits: Raw predictions [B, C, H, W] or [B, H, W]
            target: Ground truth labels [B, C, H, W] or [B, H, W]

        Returns:
            Focal loss value
        """
        # Handle different input shapes
        if logits.dim() == 3:  # [B, H, W]
            logits = logits.unsqueeze(1)  # [B, 1, H, W]
        if target.dim() == 3:  # [B, H, W]
            target = target.unsqueeze(1)  # [B, 1, H, W]

        # Flatten tensors
        logits_flat = logits.view(-1)  # [N]
        target_flat = target.view(-1)  # [N]

        # Compute probabilities
        probs = torch.sigmoid(logits_flat)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(
            logits_flat, target_flat, reduction='none'
        )

        # Compute focal weight
        p_t = probs * target_flat + (1 - probs) * (1 - target_flat)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute alpha weight
        alpha_t = self.alpha * target_flat + (1 - self.alpha) * (1 - target_flat)

        # Compute focal loss
        focal_loss = alpha_t * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss.view_as(target)


class BinaryFocalLoss(nn.Module):
    """
    Simplified Binary Focal Loss for binary segmentation.

    This is a streamlined version of Focal Loss specifically for binary
    segmentation tasks with class imbalance.

    Args:
        alpha: Balancing factor for positive class
        gamma: Focusing parameter
        smooth: Smoothing factor

    Example:
        >>> loss_fn = BinaryFocalLoss(alpha=0.8, gamma=2.0)
        >>> pred = torch.sigmoid(torch.randn(2, 1, 64, 64))
        >>> target = torch.randint(0, 2, (2, 1, 64, 64)).float()
        >>> loss = loss_fn(pred, target)
    """

    def __init__(
        self,
        alpha: float = 0.8,
        gamma: float = 2.0,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Binary Focal loss.

        Args:
            pred: Predicted probabilities [B, C, H, W] or [B, H, W]
            target: Ground truth labels [B, C, H, W] or [B, H, W]

        Returns:
            Binary Focal loss value
        """
        # Ensure pred is probability
        if pred.max() > 1.0 or pred.min() < 0.0:
            pred = torch.sigmoid(pred)

        # Handle different input shapes
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Compute focal loss components
        ce_loss = -(target_flat * torch.log(pred_flat + self.smooth) +
                   (1 - target_flat) * torch.log(1 - pred_flat + self.smooth))

        p_t = pred_flat * target_flat + (1 - pred_flat) * (1 - target_flat)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_t = self.alpha * target_flat + (1 - self.alpha) * (1 - target_flat)

        focal_loss = alpha_t * focal_weight * ce_loss

        return focal_loss.mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss combining the benefits of Focal Loss and Tversky Loss.

    This loss is particularly effective for imbalanced datasets where you want
    both the class balancing of Focal Loss and the precision/recall control
    of Tversky Loss.

    Args:
        alpha: Weight for false positives (default: 0.3)
        beta: Weight for false negatives (default: 0.7)
        gamma: Focusing parameter (default: 0.75)
        smooth: Smoothing factor (default: 1e-6)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 0.75,
        smooth: float = 1e-6
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Focal Tversky Loss.

        Args:
            pred: Predictions [N, H, W] or [N, 1, H, W]
            target: Ground truth [N, H, W] or [N, 1, H, W]

        Returns:
            Computed loss
        """
        # Ensure tensors are the right shape
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)

        # Flatten tensors
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        # Calculate Tversky index components
        tp = (pred_flat * target_flat).sum()
        fp = (pred_flat * (1 - target_flat)).sum()
        fn = ((1 - pred_flat) * target_flat).sum()

        # Calculate Tversky index
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Apply focal weight
        focal_tversky = (1 - tversky) ** self.gamma

        return focal_tversky


class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss for handling class imbalance with different penalties.

    This variant allows different focusing parameters for positive and negative
    examples, providing fine-grained control over the learning process.

    Args:
        alpha: Weight for positive class (default: 0.25)
        gamma_pos: Focusing parameter for positive examples (default: 1.0)
        gamma_neg: Focusing parameter for negative examples (default: 4.0)
        clip: Clipping value for numerical stability (default: 0.05)
        reduction: Reduction method ('mean', 'sum', 'none')
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma_pos: float = 1.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.clip = clip
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Asymmetric Focal Loss.

        Args:
            pred: Predictions [N, H, W] or [N, 1, H, W]
            target: Ground truth [N, H, W] or [N, 1, H, W]

        Returns:
            Computed loss
        """
        # Ensure tensors are the right shape
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        # Apply sigmoid and clip for numerical stability
        pred_prob = torch.sigmoid(pred)
        pred_prob = torch.clamp(pred_prob, self.clip, 1.0 - self.clip)

        # Flatten tensors
        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        # Calculate cross-entropy
        ce_loss = -(target_flat * torch.log(pred_flat) + (1 - target_flat) * torch.log(1 - pred_flat))

        # Calculate asymmetric weights
        pos_weight = self.alpha * (1 - pred_flat) ** self.gamma_pos
        neg_weight = (1 - self.alpha) * pred_flat ** self.gamma_neg

        # Apply asymmetric focal weights
        focal_weight = target_flat * pos_weight + (1 - target_flat) * neg_weight
        focal_loss = focal_weight * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


__all__ = ["FocalLoss", "BinaryFocalLoss", "FocalTverskyLoss", "AsymmetricFocalLoss"]