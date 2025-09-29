"""
Loss functions for segmentation tasks.

This module provides various loss functions commonly used in
segmentation tasks, with factory function for easy instantiation.
"""

import torch.nn as nn
from typing import Dict, Any

from .dice import DiceLoss, TverskyLoss, CompoundLoss
from .focal import FocalLoss, BinaryFocalLoss, FocalTverskyLoss, AsymmetricFocalLoss


def get_loss(loss_type: str, params: Dict[str, Any] = None) -> nn.Module:
    """
    Factory function to create loss functions.

    Args:
        loss_type: Type of loss function
        params: Parameters for the loss function

    Returns:
        Instantiated loss function

    Example:
        >>> loss_fn = get_loss("dice", {"smooth": 1.0})
        >>> loss_fn = get_loss("tversky", {"alpha": 0.3, "beta": 0.7})
        >>> loss_fn = get_loss("focal", {"alpha": 0.25, "gamma": 2.0})
    """
    if params is None:
        params = {}

    if loss_type == "dice":
        return DiceLoss(**params)
    elif loss_type == "tversky":
        return TverskyLoss(**params)
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss(**params)
    elif loss_type == "focal":
        return FocalLoss(**params)
    elif loss_type == "binary_focal":
        return BinaryFocalLoss(**params)
    elif loss_type == "focal_tversky":
        return FocalTverskyLoss(**params)
    elif loss_type == "asymmetric_focal":
        return AsymmetricFocalLoss(**params)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


__all__ = [
    "DiceLoss", "TverskyLoss", "CompoundLoss",
    "FocalLoss", "BinaryFocalLoss", "FocalTverskyLoss", "AsymmetricFocalLoss",
    "get_loss"
]