"""
Results visualization script for segmentation platform.

This script generates comprehensive visualizations for training results,
including training curves, overlays, and summary reports.
"""

import argparse
import sys
from pathlib import Path
import json
from typing import Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.config import load_config
from src.models.lightning_module import SegmentationModel
from src.data.datamodule import SegmentationDataModule
from src.visualization.overlays import generate_overlays
from src.visualization.metrics_plot import generate_all_plots
from src.utils.logging import setup_logging, get_logger


def find_best_checkpoint(run_dir: Path) -> Optional[Path]:
    """
    Find the best checkpoint in a run directory.

    Args:
        run_dir: Run directory path

    Returns:
        Path to best checkpoint or None if not found
    """
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        return None

    # Look for best.ckpt first
    best_ckpt = checkpoint_dir / "best.ckpt"
    if best_ckpt.exists():
        return best_ckpt

    # Look for any .ckpt file
    ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
    if ckpt_files:
        return ckpt_files[0]

    return None


def load_run_config(run_dir: Path) -> Optional[dict]:
    """
    Load configuration from run directory.

    Args:
        run_dir: Run directory path

    Returns:
        Configuration dictionary or None if not found
    """
    config_path = run_dir / "config.yaml"
    if config_path.exists():
        return load_config(str(config_path))
    return None


def generate_training_visualizations(
    run_dir: Path,
    output_dir: Optional[Path] = None,
    num_overlays: int = 10
) -> None:
    """
    Generate all training visualizations for a run.

    Args:
        run_dir: Run directory containing training outputs
        output_dir: Output directory for visualizations
        num_overlays: Number of overlay samples to generate
    """
    logger = get_logger(__name__)

    if output_dir is None:
        output_dir = run_dir / "visualizations"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating visualizations for run: {run_dir}")

    # Generate training curve plots
    logger.info("Generating training curves...")
    try:
        generate_all_plots(run_dir, output_dir / "plots")
    except Exception as e:
        logger.warning(f"Failed to generate training plots: {e}")

    # Generate overlay visualizations if model is available
    checkpoint_path = find_best_checkpoint(run_dir)
    config = load_run_config(run_dir)

    if checkpoint_path and config:
        logger.info("Generating overlay visualizations...")
        try:
            # Load model
            model = SegmentationModel.load_from_checkpoint(
                str(checkpoint_path),
                config=config
            )
            model.eval()

            # Create data module
            datamodule = SegmentationDataModule(config)
            datamodule.setup("fit")

            # Generate overlays
            generate_overlays(
                model=model,
                datamodule=datamodule,
                output_dir=output_dir / "overlays",
                num_samples=num_overlays
            )

        except Exception as e:
            logger.warning(f"Failed to generate overlays: {e}")
    else:
        logger.warning("No checkpoint or config found, skipping overlay generation")

    logger.info(f"Visualizations saved to: {output_dir}")


def create_run_summary(run_dir: Path, output_path: Optional[Path] = None) -> None:
    """
    Create a text summary of the training run.

    Args:
        run_dir: Run directory
        output_path: Path to save summary
    """
    if output_path is None:
        output_path = run_dir / "run_summary.txt"

    summary_lines = []
    summary_lines.append(f"Training Run Summary")
    summary_lines.append(f"{'=' * 50}")
    summary_lines.append(f"Run Directory: {run_dir}")
    summary_lines.append("")

    # Load configuration
    config = load_run_config(run_dir)
    if config:
        summary_lines.append("Configuration:")
        summary_lines.append(f"  Project: {config.get('project_name', 'N/A')}")
        summary_lines.append(f"  Model: {config.get('model', {}).get('architecture', 'N/A')}")
        summary_lines.append(f"  Encoder: {config.get('model', {}).get('encoder', 'N/A')}")
        summary_lines.append(f"  Epochs: {config.get('training', {}).get('epochs', 'N/A')}")
        summary_lines.append(f"  Batch Size: {config.get('training', {}).get('batch_size', 'N/A')}")
        summary_lines.append(f"  Learning Rate: {config.get('training', {}).get('learning_rate', 'N/A')}")
        summary_lines.append(f"  Loss: {config.get('training', {}).get('loss', {}).get('type', 'N/A')}")
        summary_lines.append("")

    # Load final metrics if available
    final_metrics_path = run_dir / "final_metrics.json"
    if final_metrics_path.exists():
        with open(final_metrics_path, 'r') as f:
            metrics = json.load(f)

        summary_lines.append("Final Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                summary_lines.append(f"  {key}: {value:.4f}")
            else:
                summary_lines.append(f"  {key}: {value}")
        summary_lines.append("")

    # Check for available outputs
    summary_lines.append("Available Outputs:")
    if (run_dir / "checkpoints").exists():
        checkpoints = list((run_dir / "checkpoints").glob("*.ckpt"))
        summary_lines.append(f"  Checkpoints: {len(checkpoints)} files")

    if (run_dir / "overlays").exists():
        overlays = list((run_dir / "overlays").glob("*.png"))
        summary_lines.append(f"  Overlays: {len(overlays)} files")

    if (run_dir / "metrics").exists():
        summary_lines.append("  Metrics: CSV logs available")

    # Dataset info
    dataset_info_path = run_dir / "dataset_info.json"
    if dataset_info_path.exists():
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)

        summary_lines.append("")
        summary_lines.append("Dataset Information:")
        summary_lines.append(f"  Training samples: {dataset_info.get('train_size', 'N/A')}")
        summary_lines.append(f"  Validation samples: {dataset_info.get('val_size', 'N/A')}")
        summary_lines.append(f"  Dataset: {dataset_info.get('dataset_name', 'N/A')}")

    # Write summary
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    logger = get_logger(__name__)
    logger.info(f"Run summary saved to: {output_path}")


def main():
    """Main visualization entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for training results",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to training run directory"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output directory for visualizations (default: run-dir/visualizations)"
    )

    parser.add_argument(
        "--num-overlays",
        type=int,
        default=10,
        help="Number of overlay samples to generate (default: 10)"
    )

    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Generate only training curve plots (skip overlays)"
    )

    parser.add_argument(
        "--overlays-only",
        action="store_true",
        help="Generate only overlay visualizations (skip plots)"
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help="Generate run summary text file"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")
    logger = get_logger(__name__)

    try:
        run_dir = Path(args.run_dir)
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")

        output_dir = Path(args.output) if args.output else run_dir / "visualizations"

        # Generate summary if requested
        if args.summary:
            create_run_summary(run_dir)

        # Generate plots only
        if args.plots_only:
            logger.info("Generating training curve plots...")
            generate_all_plots(run_dir, output_dir / "plots")

        # Generate overlays only
        elif args.overlays_only:
            logger.info("Generating overlay visualizations...")
            checkpoint_path = find_best_checkpoint(run_dir)
            config = load_run_config(run_dir)

            if not checkpoint_path or not config:
                raise ValueError("No checkpoint or config found for overlay generation")

            model = SegmentationModel.load_from_checkpoint(
                str(checkpoint_path), config=config
            )
            model.eval()

            datamodule = SegmentationDataModule(config)
            datamodule.setup("fit")

            generate_overlays(
                model=model,
                datamodule=datamodule,
                output_dir=output_dir / "overlays",
                num_samples=args.num_overlays
            )

        # Generate all visualizations
        else:
            generate_training_visualizations(
                run_dir, output_dir, args.num_overlays
            )

        logger.info("Visualization generation complete!")

    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()