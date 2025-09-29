"""
SDD v4.0 compliant Trainer wrapper.

This module provides a stable Trainer interface that wraps PyTorch Lightning
while maintaining the SDD contract stability through version changes.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from .effective_settings import EffectiveSettingsLogger
from .auto_tune import AutoTuner
from .config import ValidationMode, get_validation_mode
from ..utils.logging import get_logger

logger = get_logger(__name__)


class SegmentationTrainer:
    """
    SDD v4.0 compliant trainer wrapper.

    This class implements the stable Trainer interface contract:
    - fit(model, datamodule, callbacks=None) -> None
    - validate(model, datamodule) -> List[Dict]
    - test(model, datamodule) -> List[Dict]

    The wrapper ensures consistent behavior across PyTorch Lightning versions
    and implements SDD requirements for logging and auto-tuning.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        run_dir: Optional[Union[str, Path]] = None,
        experiment_name: Optional[str] = None
    ):
        """
        Initialize the trainer.

        Args:
            config: Configuration dictionary
            run_dir: Directory for this training run (default: config output dir)
            experiment_name: Name for logging and checkpoints
        """
        self.config = config
        self.run_dir = Path(run_dir) if run_dir else Path(config.get("output", {}).get("dir", "./runs"))
        self.experiment_name = experiment_name or "segmentation_experiment"

        # Initialize effective settings logger
        self.settings_logger = EffectiveSettingsLogger(run_dir)

        # Initialize auto-tuner (may be disabled)
        self.auto_tuner = AutoTuner(config, self.settings_logger)

        # Store the actual PyTorch Lightning trainer
        self._trainer: Optional[pl.Trainer] = None

    def fit(
        self,
        model,
        datamodule,
        callbacks: Optional[List] = None
    ) -> None:
        """
        Train the model.

        Args:
            model: Lightning module to train
            datamodule: Data module providing train/val data
            callbacks: Optional additional callbacks
        """
        trainer = self._create_trainer(callbacks=callbacks)
        trainer.fit(model, datamodule)

        # Save effective settings after training
        self.settings_logger.save()

    def validate(self, model, datamodule) -> List[Dict]:
        """
        Validate the model.

        Args:
            model: Lightning module to validate
            datamodule: Data module providing validation data

        Returns:
            List of validation metrics dictionaries
        """
        trainer = self._create_trainer(mode='validate')
        results = trainer.validate(model, datamodule)
        return results

    def test(self, model, datamodule) -> List[Dict]:
        """
        Test the model.

        Args:
            model: Lightning module to test
            datamodule: Data module providing test data

        Returns:
            List of test metrics dictionaries
        """
        trainer = self._create_trainer(mode='test')
        results = trainer.test(model, datamodule)
        return results

    def _create_trainer(
        self,
        callbacks: Optional[List] = None,
        mode: str = 'fit'
    ) -> pl.Trainer:
        """
        Create PyTorch Lightning trainer with SDD compliance.

        Args:
            callbacks: Optional callbacks to add
            mode: Mode for trainer ('fit', 'validate', 'test')

        Returns:
            Configured PyTorch Lightning trainer
        """
        # Get training configuration
        training_config = self.config.get('training', {})
        compute_config = self.config.get('compute', {})

        # Apply auto-tuning if enabled
        epochs = training_config.get('epochs', 50)
        batch_size = training_config.get('batch_size', 8)
        precision = compute_config.get('precision', 32)

        # Auto-tune settings if requested
        effective_batch_size = self.auto_tuner.tune_batch_size(batch_size)
        effective_precision = self.auto_tuner.tune_precision(precision)

        # Log non-auto-tuned settings
        self.settings_logger.log_setting('epochs', epochs, epochs)
        self.settings_logger.log_setting('accelerator',
                                        compute_config.get('accelerator', 'auto'),
                                        compute_config.get('accelerator', 'auto'))

        # Setup callbacks
        trainer_callbacks = []

        if mode == 'fit':
            # Add checkpointing
            checkpoint_config = self.config.get('output', {}).get('checkpoint', {})
            checkpoint_callback = ModelCheckpoint(
                dirpath=self.run_dir / "checkpoints",
                monitor=checkpoint_config.get('monitor', 'val_loss'),
                mode=checkpoint_config.get('mode', 'min'),
                save_top_k=1 if checkpoint_config.get('save_best_only', True) else -1,
                filename='best'
            )
            trainer_callbacks.append(checkpoint_callback)

            # Add early stopping if configured
            if 'early_stopping' in training_config:
                early_stop_config = training_config['early_stopping']
                early_stopping = EarlyStopping(
                    monitor=early_stop_config.get('monitor', 'val_loss'),
                    patience=early_stop_config.get('patience', 10),
                    mode=early_stop_config.get('mode', 'min')
                )
                trainer_callbacks.append(early_stopping)

        # Add user callbacks
        if callbacks:
            trainer_callbacks.extend(callbacks)

        # Setup logger
        tb_logger = TensorBoardLogger(
            save_dir=self.run_dir,
            name="lightning_logs"
        )

        # Create trainer
        trainer_kwargs = {
            'max_epochs': epochs,
            'accelerator': compute_config.get('accelerator', 'auto'),
            'devices': compute_config.get('devices', 1),
            'precision': effective_precision,
            'deterministic': compute_config.get('deterministic', True),
            'callbacks': trainer_callbacks,
            'logger': tb_logger,
            'default_root_dir': str(self.run_dir)
        }

        # Add fast_dev_run if in config
        if self.config.get('fast_dev_run', False):
            trainer_kwargs['fast_dev_run'] = True

        # Validate configuration based on mode
        validation_mode = get_validation_mode()
        if validation_mode == ValidationMode.STRICT:
            # In strict mode, ensure all required fields are present
            required_fields = ['epochs', 'accelerator']
            for field in required_fields:
                if field not in trainer_kwargs or trainer_kwargs[field] is None:
                    raise ValueError(f"Required field '{field}' missing in STRICT validation mode")

        self._trainer = pl.Trainer(**trainer_kwargs)
        return self._trainer

    @property
    def lightning_trainer(self) -> Optional[pl.Trainer]:
        """
        Get the underlying PyTorch Lightning trainer.

        Returns:
            The PyTorch Lightning trainer instance (if created)
        """
        return self._trainer


    @property
    def logged_metrics(self) -> Dict[str, Any]:
        """Expose logged metrics from the underlying Lightning trainer."""
        if self._trainer is None:
            return {}
        return getattr(self._trainer, "logged_metrics", {})


__all__ = ["SegmentationTrainer"]
