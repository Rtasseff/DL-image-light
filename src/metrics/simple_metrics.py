"""
Simple segmentation metrics for testing without external dependencies.

This module provides basic implementations for SDD-required metrics
that can work without torchmetrics dependency for validation.
"""

import torch


def compute_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union (IoU).

    Args:
        pred: Predictions [N, H, W] or [N, 1, H, W]
        target: Ground truth [N, H, W] or [N, 1, H, W]
        threshold: Threshold for binarizing predictions

    Returns:
        IoU value
    """
    # Ensure same shape
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    # Compute IoU
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return (intersection / union).item()


def compute_precision(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Precision.

    Args:
        pred: Predictions [N, H, W] or [N, 1, H, W]
        target: Ground truth [N, H, W] or [N, 1, H, W]
        threshold: Threshold for binarizing predictions

    Returns:
        Precision value
    """
    # Ensure same shape
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    # Compute precision
    tp = (pred_binary * target_binary).sum()
    fp = (pred_binary * (1 - target_binary)).sum()

    if tp + fp == 0:
        return 0.0

    return (tp / (tp + fp)).item()


def compute_recall(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Recall.

    Args:
        pred: Predictions [N, H, W] or [N, 1, H, W]
        target: Ground truth [N, H, W] or [N, 1, H, W]
        threshold: Threshold for binarizing predictions

    Returns:
        Recall value
    """
    # Ensure same shape
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    # Compute recall
    tp = (pred_binary * target_binary).sum()
    fn = ((1 - pred_binary) * target_binary).sum()

    if tp + fn == 0:
        return 0.0

    return (tp / (tp + fn)).item()


def compute_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Compute Dice coefficient.

    Args:
        pred: Predictions [N, H, W] or [N, 1, H, W]
        target: Ground truth [N, H, W] or [N, 1, H, W]
        threshold: Threshold for binarizing predictions

    Returns:
        Dice value
    """
    # Ensure same shape
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # Binarize
    pred_binary = (pred > threshold).float()
    target_binary = target.float()

    # Compute Dice
    intersection = (pred_binary * target_binary).sum()
    total = pred_binary.sum() + target_binary.sum()

    if total == 0:
        return 1.0 if intersection == 0 else 0.0

    return (2.0 * intersection / total).item()


__all__ = ["compute_iou", "compute_precision", "compute_recall", "compute_dice"]