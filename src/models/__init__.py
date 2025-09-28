"""
Models module for segmentation platform.

This module provides the Lightning model implementations
and factory functions for creating models.
"""

from .lightning_module import SegmentationModel

__all__ = ["SegmentationModel"]