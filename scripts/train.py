"""
Main training script for segmentation platform.

This script orchestrates the entire training pipeline from config
loading through model training to results visualization.
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

import yaml
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.core.config import load_and_validate_config
from src.data.datamodule import SegmentationDataModule
from src.models.lightning_module import SegmentationModel
from src.utils.reproducibility import set_global_seed
from src.utils.logging import setup_logging, get_logger


def save_environment_info(output_path: Path) -> None:
    """Save environment information for reproducibility."""
    import torch
    import torchvision
    import segmentation_models_pytorch as smp
    import pytorch_lightning as pl_version
    import albumentations as A

    env_info = {
        "python": sys.version,
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "lightning": pl_version.__version__,
        "smp": smp.__version__,
        "albumentations": A.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
    }

    with open(output_path, "w") as f:
        json.dump(env_info, f, indent=2)


def create_callbacks(config: Dict[str, Any], run_dir: Path) -> list:
    """
    Create Lightning callbacks from configuration.

    Args:
        config: Configuration dictionary
        run_dir: Run directory for outputs

    Returns:
        List of callbacks
    """
    callbacks = []

    # Model checkpoint callback
    checkpoint_config = config["output"]["checkpoint"]
    callbacks.append(
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best",
            monitor=checkpoint_config["monitor"],
            mode=checkpoint_config["mode"],
            save_top_k=1,
            verbose=True
        )
    )

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    # Early stopping if configured
    if "early_stopping" in config["training"]:
        early_stopping_config = config["training"]["early_stopping"]
        callbacks.append(
            EarlyStopping(
                monitor=early_stopping_config["monitor"],
                patience=early_stopping_config["patience"],
                mode=early_stopping_config["mode"],
                verbose=True
            )
        )

    return callbacks


def create_trainer(config: Dict[str, Any], run_dir: Path, callbacks: list) -> pl.Trainer:
    """
    Create Lightning trainer from configuration.

    Args:
        config: Configuration dictionary
        run_dir: Run directory for outputs
        callbacks: List of callbacks

    Returns:
        Configured trainer
    """
    compute_config = config["compute"]

    # Setup logger
    csv_logger = CSVLogger(run_dir / "metrics", name="", version="")

    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=config["training"]["epochs"],
        accelerator=compute_config["accelerator"],
        devices=compute_config["devices"],
        precision=str(compute_config["precision"]),  # Convert to string for Lightning 2.0+
        deterministic=compute_config["deterministic"],
        callbacks=callbacks,
        logger=csv_logger,
        enable_progress_bar=True,
        log_every_n_steps=10,
        accumulate_grad_batches=config["training"].get("accumulate_grad_batches", 1),
        enable_checkpointing=True,
        enable_model_summary=True,
    )

    return trainer


def main():
    """Main training entry point."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Train segmentation model",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Experiment name (defaults to config name + timestamp)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training"
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run one batch for debugging"
    )
    args = parser.parse_args()

    try:
        # Load and validate configuration
        print(f"Loading configuration from: {args.config}")
        config = load_and_validate_config(args.config)

        # Setup experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = Path(args.config).stem
        exp_name = args.name or f"{config_name}_{timestamp}"
        run_dir = Path(config["output"]["dir"]) / exp_name
        run_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        setup_logging(
            run_dir / "train.log",
            level=config["logging"]["level"]
        )
        logger = get_logger(__name__)
        logger.info(f"Starting training: {exp_name}")
        logger.info(f"Run directory: {run_dir}")

        # Save configuration and environment info
        with open(run_dir / "config.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        save_environment_info(run_dir / "environment.json")

        # Set random seeds for reproducibility
        seed = config["compute"]["seed"]
        set_global_seed(seed)
        logger.info(f"Random seed set to: {seed}")

        # Create data module
        logger.info("Setting up data module...")
        datamodule = SegmentationDataModule(config)
        datamodule.setup("fit")

        # Log data statistics
        dataset_info = datamodule.get_dataset_info()
        logger.info(f"Training samples: {dataset_info['train_size']}")
        logger.info(f"Validation samples: {dataset_info['val_size']}")

        # Save dataset info
        with open(run_dir / "dataset_info.json", "w") as f:
            json.dump(dataset_info, f, indent=2)

        # Create model
        logger.info("Creating model...")
        model = SegmentationModel(config)
        logger.info(model.get_model_summary())

        # Create callbacks
        callbacks = create_callbacks(config, run_dir)

        # Create trainer
        logger.info("Initializing trainer...")
        trainer = create_trainer(config, run_dir, callbacks)

        # Override for fast dev run
        if args.fast_dev_run:
            trainer.fast_dev_run = True
            logger.info("Running in fast dev run mode (1 batch)")

        # Train model
        logger.info("Starting training...")
        trainer.fit(
            model=model,
            datamodule=datamodule,
            ckpt_path=args.resume
        )

        # Run final validation
        if not args.fast_dev_run:
            logger.info("Running final validation...")
            trainer.validate(
                model=model,
                datamodule=datamodule,
                ckpt_path="best"
            )

        logger.info(f"Training complete! Results saved to: {run_dir}")

        # Print final metrics
        if trainer.logged_metrics:
            logger.info("Final metrics:")
            for key, value in trainer.logged_metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {key}: {value:.4f}")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()