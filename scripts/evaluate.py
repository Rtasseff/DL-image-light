"""
Evaluation script for segmentation platform.

This script evaluates a trained model on a test dataset,
computing comprehensive metrics and generating evaluation reports.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.lightning_module import SegmentationModel
from src.core.config import load_config
from src.data.datamodule import SegmentationDataModule
from src.metrics import get_metric
from src.visualization.overlays import create_error_visualization, generate_overlays
from src.visualization.metrics_plot import create_confusion_matrix_plot
from src.utils.logging import setup_logging, get_logger


def compute_pixel_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute pixel-level metrics for binary segmentation.

    Args:
        predictions: Predicted probabilities [N, H, W]
        targets: Ground truth masks [N, H, W]
        threshold: Threshold for binarizing predictions

    Returns:
        Dictionary of computed metrics
    """
    # Binarize predictions
    pred_binary = (predictions > threshold).float()
    targets_binary = targets.float()

    # Flatten for computation
    pred_flat = pred_binary.view(-1)
    target_flat = targets_binary.view(-1)

    # Compute confusion matrix components
    tp = (pred_flat * target_flat).sum().item()
    fp = (pred_flat * (1 - target_flat)).sum().item()
    fn = ((1 - pred_flat) * target_flat).sum().item()
    tn = ((1 - pred_flat) * (1 - target_flat)).sum().item()

    # Compute metrics
    metrics = {}

    # Basic metrics
    metrics['true_positives'] = tp
    metrics['false_positives'] = fp
    metrics['false_negatives'] = fn
    metrics['true_negatives'] = tn

    # Derived metrics
    metrics['accuracy'] = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # F1 and Dice scores
    metrics['f1_score'] = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    metrics['dice_score'] = metrics['f1_score']  # Same as F1 for binary case

    # IoU
    metrics['iou'] = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return metrics


def evaluate_model_on_dataloader(
    model: SegmentationModel,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[Dict[str, float], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Evaluate model on a dataloader.

    Args:
        model: Trained segmentation model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        threshold: Threshold for predictions

    Returns:
        Tuple of (metrics_dict, predictions_list, targets_list, images_list)
    """
    model.eval()

    all_predictions = []
    all_targets = []
    all_images = []

    logger = get_logger(__name__)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Evaluating")):
            images = images.to(device)
            targets = targets.to(device)

            # Get predictions
            logits = model(images)
            predictions = torch.sigmoid(logits)

            # Store for later analysis
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_images.append(images.cpu())

    # Concatenate all results
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_images = torch.cat(all_images, dim=0)

    # Remove channel dimension if present
    if all_predictions.dim() == 4:
        all_predictions = all_predictions.squeeze(1)
    if all_targets.dim() == 4:
        all_targets = all_targets.squeeze(1)

    # Compute overall metrics
    metrics = compute_pixel_metrics(all_predictions, all_targets, threshold)

    # Convert to lists for return
    pred_list = [pred.numpy() for pred in all_predictions]
    target_list = [target.numpy() for target in all_targets]
    image_list = [img.numpy() for img in all_images]

    return metrics, pred_list, target_list, image_list


def compute_per_image_metrics(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    threshold: float = 0.5
) -> pd.DataFrame:
    """
    Compute metrics for each image individually.

    Args:
        predictions: List of prediction arrays
        targets: List of target arrays
        threshold: Threshold for binarizing predictions

    Returns:
        DataFrame with per-image metrics
    """
    per_image_metrics = []

    for i, (pred, target) in enumerate(zip(predictions, targets)):
        pred_tensor = torch.from_numpy(pred).unsqueeze(0)
        target_tensor = torch.from_numpy(target).unsqueeze(0)

        metrics = compute_pixel_metrics(pred_tensor, target_tensor, threshold)
        metrics['image_id'] = i
        per_image_metrics.append(metrics)

    return pd.DataFrame(per_image_metrics)


def save_evaluation_results(
    overall_metrics: Dict[str, float],
    per_image_metrics: pd.DataFrame,
    output_dir: Path,
    threshold: float
) -> None:
    """
    Save evaluation results to files.

    Args:
        overall_metrics: Overall dataset metrics
        per_image_metrics: Per-image metrics DataFrame
        output_dir: Output directory
        threshold: Threshold used for evaluation
    """
    # Save overall metrics
    overall_path = output_dir / "overall_metrics.json"
    with open(overall_path, 'w') as f:
        json.dump({
            'threshold': threshold,
            'metrics': overall_metrics,
            'summary': {
                'dice_score': overall_metrics['dice_score'],
                'iou': overall_metrics['iou'],
                'precision': overall_metrics['precision'],
                'recall': overall_metrics['recall'],
                'accuracy': overall_metrics['accuracy']
            }
        }, f, indent=2)

    # Save per-image metrics
    per_image_path = output_dir / "per_image_metrics.csv"
    per_image_metrics.to_csv(per_image_path, index=False)

    # Save summary statistics
    summary_stats = per_image_metrics[['dice_score', 'iou', 'precision', 'recall', 'accuracy']].describe()
    summary_path = output_dir / "metrics_summary.csv"
    summary_stats.to_csv(summary_path)

    logger = get_logger(__name__)
    logger.info(f"Evaluation results saved to {output_dir}")


def generate_evaluation_visualizations(
    images: List[np.ndarray],
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    output_dir: Path,
    num_samples: int = 10,
    threshold: float = 0.5
) -> None:
    """
    Generate evaluation visualizations.

    Args:
        images: List of input images
        predictions: List of predictions
        targets: List of ground truth masks
        output_dir: Output directory
        num_samples: Number of samples to visualize
        threshold: Threshold for predictions
    """
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    logger = get_logger(__name__)
    logger.info(f"Generating {num_samples} evaluation visualizations...")

    for i in range(min(num_samples, len(images))):
        # Denormalize image
        image = images[i].transpose(1, 2, 0)  # CHW -> HWC
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)

        prediction = predictions[i]
        target = targets[i]

        # Create error visualization
        error_vis = create_error_visualization(image, prediction, target, threshold)

        # Save error visualization
        error_path = vis_dir / f"error_analysis_{i:03d}.png"
        from PIL import Image as PILImage
        PILImage.fromarray(error_vis).save(error_path)

    logger.info(f"Visualizations saved to {vis_dir}")


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.ckpt file)"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )

    parser.add_argument(
        "--test-dir",
        type=str,
        help="Path to test dataset directory (if different from training)"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for evaluation results"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary predictions (default: 0.5)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, mps, auto)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)"
    )

    parser.add_argument(
        "--num-vis",
        type=int,
        default=10,
        help="Number of samples to visualize (default: 10)"
    )

    parser.add_argument(
        "--use-val",
        action="store_true",
        help="Evaluate on validation set instead of test set"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    try:
        # Determine device
        if args.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        else:
            device = torch.device(args.device)

        logger.info(f"Using device: {device}")

        # Load configuration
        config = load_config(args.config)

        # Override batch size
        config["training"]["batch_size"] = args.batch_size

        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        model = SegmentationModel.load_from_checkpoint(args.checkpoint, config=config)
        model.to(device)

        # Create data module
        if args.test_dir:
            # Use custom test directory
            config["dataset"]["images_dir"] = args.test_dir + "/images"
            config["dataset"]["masks_dir"] = args.test_dir + "/masks"

        datamodule = SegmentationDataModule(config)

        if args.use_val:
            datamodule.setup("fit")
            eval_loader = datamodule.val_dataloader()
            eval_split = "validation"
        else:
            datamodule.setup("test")
            eval_loader = datamodule.test_dataloader()
            if eval_loader is None:
                # Fallback to validation if no test set
                datamodule.setup("fit")
                eval_loader = datamodule.val_dataloader()
                eval_split = "validation"
                logger.warning("No test set available, using validation set")
            else:
                eval_split = "test"

        logger.info(f"Evaluating on {eval_split} set")

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run evaluation
        logger.info("Starting evaluation...")
        overall_metrics, predictions, targets, images = evaluate_model_on_dataloader(
            model, eval_loader, device, args.threshold
        )

        # Compute per-image metrics
        logger.info("Computing per-image metrics...")
        per_image_metrics = compute_per_image_metrics(predictions, targets, args.threshold)

        # Save results
        save_evaluation_results(overall_metrics, per_image_metrics, output_dir, args.threshold)

        # Generate visualizations
        if args.num_vis > 0:
            generate_evaluation_visualizations(
                images, predictions, targets, output_dir, args.num_vis, args.threshold
            )

        # Generate confusion matrix plot
        create_confusion_matrix_plot(
            overall_metrics['true_positives'],
            overall_metrics['false_positives'],
            overall_metrics['false_negatives'],
            overall_metrics['true_negatives'],
            output_dir / "confusion_matrix.png"
        )

        # Print summary
        logger.info("Evaluation Summary:")
        logger.info(f"  Dice Score: {overall_metrics['dice_score']:.4f}")
        logger.info(f"  IoU:        {overall_metrics['iou']:.4f}")
        logger.info(f"  Precision:  {overall_metrics['precision']:.4f}")
        logger.info(f"  Recall:     {overall_metrics['recall']:.4f}")
        logger.info(f"  Accuracy:   {overall_metrics['accuracy']:.4f}")
        logger.info(f"  F1 Score:   {overall_metrics['f1_score']:.4f}")

        logger.info(f"Full results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()