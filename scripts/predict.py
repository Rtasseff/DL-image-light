"""
Prediction script for segmentation platform.

This script performs inference on new images using a trained model,
generating predictions and optional visualizations.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.models.lightning_module import SegmentationModel
from src.core.config import load_config
from src.visualization.overlays import create_overlay, create_comparison_grid
from src.utils.logging import setup_logging, get_logger


def load_model_from_checkpoint(checkpoint_path: str, config_path: str) -> SegmentationModel:
    """
    Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to training configuration

    Returns:
        Loaded model
    """
    config = load_config(config_path)
    model = SegmentationModel.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()
    return model


def create_inference_transform(image_size: Optional[tuple] = None) -> A.Compose:
    """
    Create transform pipeline for inference.

    Args:
        image_size: Target image size (height, width)

    Returns:
        Albumentations transform pipeline
    """
    transforms = []

    if image_size:
        transforms.append(A.Resize(*image_size))

    transforms.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0
        ),
        ToTensorV2()
    ])

    return A.Compose(transforms)


def load_and_preprocess_image(
    image_path: str,
    transform: A.Compose
) -> tuple:
    """
    Load and preprocess image for inference.

    Args:
        image_path: Path to input image
        transform: Preprocessing transform

    Returns:
        Tuple of (preprocessed_tensor, original_image_array)
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    # Apply transforms
    transformed = transform(image=image_array)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension

    return image_tensor, image_array


def predict_single_image(
    model: SegmentationModel,
    image_tensor: torch.Tensor,
    device: torch.device
) -> np.ndarray:
    """
    Predict on a single image.

    Args:
        model: Trained segmentation model
        image_tensor: Preprocessed image tensor [1, C, H, W]
        device: Device to run inference on

    Returns:
        Prediction mask as numpy array [H, W]
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        logits = model(image_tensor)
        probabilities = torch.sigmoid(logits)
        prediction = probabilities[0, 0].cpu().numpy()

    return prediction


def save_prediction(
    prediction: np.ndarray,
    output_path: str,
    threshold: float = 0.5
) -> None:
    """
    Save prediction as binary mask.

    Args:
        prediction: Prediction probabilities [H, W]
        output_path: Path to save prediction
        threshold: Threshold for binarization
    """
    binary_mask = (prediction > threshold).astype(np.uint8) * 255
    Image.fromarray(binary_mask, mode='L').save(output_path)


def process_image_list(
    image_paths: List[str],
    model: SegmentationModel,
    output_dir: str,
    transform: A.Compose,
    device: torch.device,
    save_overlays: bool = True,
    save_raw: bool = True,
    threshold: float = 0.5
) -> None:
    """
    Process a list of images for prediction.

    Args:
        image_paths: List of image file paths
        model: Trained model
        output_dir: Output directory
        transform: Preprocessing transform
        device: Device for inference
        save_overlays: Whether to save overlay visualizations
        save_raw: Whether to save raw probability maps
        threshold: Threshold for binary masks
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_overlays:
        overlay_dir = output_dir / "overlays"
        overlay_dir.mkdir(exist_ok=True)

    if save_raw:
        raw_dir = output_dir / "raw_predictions"
        raw_dir.mkdir(exist_ok=True)

    binary_dir = output_dir / "binary_masks"
    binary_dir.mkdir(exist_ok=True)

    logger = get_logger(__name__)
    logger.info(f"Processing {len(image_paths)} images...")

    for i, image_path in enumerate(image_paths):
        logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")

        try:
            # Load and preprocess
            image_tensor, original_image = load_and_preprocess_image(
                image_path, transform
            )

            # Predict
            prediction = predict_single_image(model, image_tensor, device)

            # Get base filename
            base_name = Path(image_path).stem

            # Save binary mask
            binary_path = binary_dir / f"{base_name}_mask.png"
            save_prediction(prediction, binary_path, threshold)

            # Save raw probabilities if requested
            if save_raw:
                raw_path = raw_dir / f"{base_name}_prob.npy"
                np.save(raw_path, prediction)

            # Save overlay if requested
            if save_overlays:
                overlay = create_overlay(original_image, prediction)
                overlay_path = overlay_dir / f"{base_name}_overlay.png"
                Image.fromarray(overlay).save(overlay_path)

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            continue

    logger.info(f"Prediction complete. Results saved to {output_dir}")


def main():
    """Main prediction entry point."""
    parser = argparse.ArgumentParser(
        description="Run inference on images using trained segmentation model",
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
        "--input",
        type=str,
        required=True,
        help="Path to input image or directory of images"
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for predictions"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary mask generation (default: 0.5)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, mps, auto)"
    )

    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        default=None,
        metavar=("HEIGHT", "WIDTH"),
        help="Resize images to this size before prediction"
    )

    parser.add_argument(
        "--no-overlays",
        action="store_true",
        help="Skip generating overlay visualizations"
    )

    parser.add_argument(
        "--no-raw",
        action="store_true",
        help="Skip saving raw probability maps"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
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

        # Load model
        logger.info(f"Loading model from {args.checkpoint}")
        model = load_model_from_checkpoint(args.checkpoint, args.config)
        model.to(device)

        # Create transform
        transform = create_inference_transform(
            image_size=tuple(args.image_size) if args.image_size else None
        )

        # Get input images
        input_path = Path(args.input)
        if input_path.is_file():
            # Single image
            image_paths = [str(input_path)]
        elif input_path.is_dir():
            # Directory of images
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            image_paths = []
            for ext in extensions:
                image_paths.extend(input_path.glob(f"*{ext}"))
                image_paths.extend(input_path.glob(f"*{ext.upper()}"))
            image_paths = [str(p) for p in image_paths]
        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")

        if not image_paths:
            raise ValueError(f"No images found in {input_path}")

        logger.info(f"Found {len(image_paths)} images to process")

        # Process images
        process_image_list(
            image_paths=image_paths,
            model=model,
            output_dir=args.output,
            transform=transform,
            device=device,
            save_overlays=not args.no_overlays,
            save_raw=not args.no_raw,
            threshold=args.threshold
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()