"""
Loss functions for segmentation tasks.

This module provides various loss functions commonly used in
segmentation tasks, with factory function for easy instantiation.
"""

import torch.nn as nn
from typing import Dict, Any

from .dice import DiceLoss, TverskyLoss, CompoundLoss


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
        # Placeholder for future implementation
        raise NotImplementedError("Focal loss not implemented yet")
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


__all__ = ["DiceLoss", "TverskyLoss", "CompoundLoss", "get_loss"]