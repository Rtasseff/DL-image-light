"""
Visualization utilities for generating prediction overlays.

This module provides functions to create visual overlays of predictions
on original images for qualitative evaluation of segmentation results.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Union, Optional, Tuple, List
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from PIL import Image
import cv2

from ..utils.logging import get_logger

logger = get_logger(__name__)


def apply_colormap(mask: np.ndarray, colormap: str = "jet", alpha: float = 0.5) -> np.ndarray:
    """
    Apply colormap to a binary or grayscale mask.

    Args:
        mask: Binary or grayscale mask [H, W]
        colormap: Matplotlib colormap name
        alpha: Transparency for the overlay

    Returns:
        Colored mask [H, W, 4] with RGBA channels
    """
    # Normalize mask to [0, 1]
    if mask.max() > 1.0:
        mask = mask.astype(np.float32) / 255.0

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored_mask = cmap(mask)

    # Set alpha channel
    colored_mask[..., 3] = alpha * mask

    return colored_mask


def create_overlay(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    pred_color: str = "red",
    gt_color: str = "green",
    alpha: float = 0.6
) -> np.ndarray:
    """
    Create overlay visualization of predictions on original image.

    Args:
        image: Original image [H, W, 3] in RGB format
        prediction: Predicted mask [H, W] with values in [0, 1]
        ground_truth: Optional ground truth mask [H, W]
        pred_color: Color for prediction overlay
        gt_color: Color for ground truth overlay
        alpha: Transparency of overlays

    Returns:
        Overlay image [H, W, 3] in RGB format
    """
    # Ensure image is in proper format
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)

    # Convert to float for blending
    overlay = image.astype(np.float32) / 255.0

    # Create prediction overlay
    if prediction is not None:
        pred_mask = (prediction > 0.5).astype(np.float32)

        # Create colored overlay
        if pred_color == "red":
            color_overlay = np.zeros_like(overlay)
            color_overlay[..., 0] = pred_mask  # Red channel
        elif pred_color == "blue":
            color_overlay = np.zeros_like(overlay)
            color_overlay[..., 2] = pred_mask  # Blue channel
        elif pred_color == "green":
            color_overlay = np.zeros_like(overlay)
            color_overlay[..., 1] = pred_mask  # Green channel
        else:
            # Use matplotlib color
            rgb = mcolors.to_rgb(pred_color)
            color_overlay = np.zeros_like(overlay)
            for i, c in enumerate(rgb):
                color_overlay[..., i] = pred_mask * c

        # Blend with original image
        overlay = overlay * (1 - alpha * pred_mask[..., None]) + color_overlay * alpha

    # Add ground truth overlay if provided
    if ground_truth is not None:
        gt_mask = (ground_truth > 0.5).astype(np.float32)

        # Create ground truth overlay (usually green)
        if gt_color == "green":
            gt_overlay = np.zeros_like(overlay)
            gt_overlay[..., 1] = gt_mask  # Green channel
        elif gt_color == "blue":
            gt_overlay = np.zeros_like(overlay)
            gt_overlay[..., 2] = gt_mask  # Blue channel
        else:
            rgb = mcolors.to_rgb(gt_color)
            gt_overlay = np.zeros_like(overlay)
            for i, c in enumerate(rgb):
                gt_overlay[..., i] = gt_mask * c

        # Blend ground truth with lower alpha to show both
        gt_alpha = alpha * 0.7
        overlay = overlay * (1 - gt_alpha * gt_mask[..., None]) + gt_overlay * gt_alpha

    # Convert back to uint8
    overlay = (overlay * 255).clip(0, 255).astype(np.uint8)

    return overlay


def create_comparison_grid(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: Optional[np.ndarray] = None,
    titles: Optional[List[str]] = None
) -> np.ndarray:
    """
    Create a comparison grid showing original, prediction, ground truth, and overlay.

    Args:
        image: Original image [H, W, 3]
        prediction: Predicted mask [H, W]
        ground_truth: Optional ground truth mask [H, W]
        titles: Optional titles for subplots

    Returns:
        Grid image as numpy array
    """
    # Setup figure
    if ground_truth is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Default titles
    if titles is None:
        if ground_truth is not None:
            titles = ["Original", "Ground Truth", "Prediction", "Overlay"]
        else:
            titles = ["Original", "Prediction", "Overlay"]

    # Original image
    axes[0].imshow(image)
    axes[0].set_title(titles[0])
    axes[0].axis('off')

    if ground_truth is not None:
        # Ground truth
        axes[1].imshow(ground_truth, cmap='gray')
        axes[1].set_title(titles[1])
        axes[1].axis('off')

        # Prediction
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title(titles[2])
        axes[2].axis('off')

        # Overlay
        overlay = create_overlay(image, prediction, ground_truth)
        axes[3].imshow(overlay)
        axes[3].set_title(titles[3])
        axes[3].axis('off')
    else:
        # Prediction
        axes[1].imshow(prediction, cmap='gray')
        axes[1].set_title(titles[1])
        axes[1].axis('off')

        # Overlay
        overlay = create_overlay(image, prediction)
        axes[2].imshow(overlay)
        axes[2].set_title(titles[2])
        axes[2].axis('off')

    plt.tight_layout()

    # Convert to numpy array
    fig.canvas.draw()
    grid_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    grid_array = grid_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return grid_array


def generate_overlays(
    model,
    datamodule,
    output_dir: Union[str, Path],
    num_samples: int = 10,
    device: Optional[torch.device] = None
) -> None:
    """
    Generate overlay visualizations for validation samples.

    Args:
        model: Trained segmentation model
        datamodule: Lightning data module
        output_dir: Directory to save overlays
        num_samples: Number of samples to visualize
        device: Device to run inference on
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Get validation dataloader
    val_loader = datamodule.val_dataloader()

    logger.info(f"Generating {num_samples} overlay visualizations...")

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader):
            if batch_idx >= num_samples:
                break

            # Move to device
            images = images.to(device)
            masks = masks.to(device)

            # Get predictions
            logits = model(images)
            predictions = torch.sigmoid(logits)

            # Process each image in batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                if batch_idx * batch_size + i >= num_samples:
                    break

                # Convert to numpy
                image_np = images[i].cpu().permute(1, 2, 0).numpy()
                pred_np = predictions[i, 0].cpu().numpy()
                mask_np = masks[i, 0].cpu().numpy()

                # Denormalize image (assuming ImageNet normalization)
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                image_np = image_np * std + mean
                image_np = np.clip(image_np, 0, 1)

                # Create comparison grid
                grid = create_comparison_grid(
                    image_np,
                    pred_np,
                    mask_np,
                    titles=["Original", "Ground Truth", "Prediction", "Overlay"]
                )

                # Save grid
                sample_idx = batch_idx * batch_size + i
                output_path = output_dir / f"overlay_{sample_idx:03d}.png"
                Image.fromarray(grid).save(output_path)

                # Also save individual overlay
                overlay = create_overlay(image_np, pred_np, mask_np)
                overlay_path = output_dir / f"overlay_only_{sample_idx:03d}.png"
                Image.fromarray(overlay).save(overlay_path)

    logger.info(f"Overlays saved to {output_dir}")


def create_error_visualization(
    image: np.ndarray,
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Create error visualization showing true positives, false positives, and false negatives.

    Args:
        image: Original image [H, W, 3]
        prediction: Predicted mask [H, W]
        ground_truth: Ground truth mask [H, W]
        threshold: Threshold for binarizing predictions

    Returns:
        Error visualization [H, W, 3]
    """
    # Binarize predictions and ground truth
    pred_binary = (prediction > threshold).astype(np.uint8)
    gt_binary = (ground_truth > threshold).astype(np.uint8)

    # Calculate error types
    true_positive = (pred_binary == 1) & (gt_binary == 1)
    false_positive = (pred_binary == 1) & (gt_binary == 0)
    false_negative = (pred_binary == 0) & (gt_binary == 1)

    # Create error overlay
    error_overlay = np.zeros_like(image, dtype=np.float32)

    # True positives (green)
    error_overlay[true_positive, 1] = 1.0

    # False positives (red)
    error_overlay[false_positive, 0] = 1.0

    # False negatives (blue)
    error_overlay[false_negative, 2] = 1.0

    # Blend with original image
    if image.max() <= 1.0:
        image_norm = image.astype(np.float32)
    else:
        image_norm = image.astype(np.float32) / 255.0

    alpha = 0.6
    blended = image_norm * (1 - alpha) + error_overlay * alpha

    return (blended * 255).clip(0, 255).astype(np.uint8)


__all__ = [
    "apply_colormap",
    "create_overlay",
    "create_comparison_grid",
    "generate_overlays",
    "create_error_visualization"
]