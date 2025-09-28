"""
Integration tests for the training pipeline.

These tests verify that the complete training pipeline works end-to-end.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import pytorch_lightning as pl

from src.core.config import load_config
from src.data.datamodule import SegmentationDataModule
from src.models.lightning_module import SegmentationModel
from src.utils.reproducibility import set_global_seed


class TestTrainingPipeline:
    """Test the complete training pipeline."""

    def test_single_epoch_training(self, test_config_file, temp_dir):
        """Test that training runs for a single epoch without errors."""
        # Load configuration
        config = load_config(test_config_file)

        # Set output directory to temp
        config["output"]["dir"] = str(temp_dir / "runs")

        # Set reproducibility
        set_global_seed(42)

        # Create data module
        datamodule = SegmentationDataModule(config)
        datamodule.setup("fit")

        # Verify data loading works
        assert len(datamodule.train_dataset) > 0
        assert len(datamodule.val_dataset) > 0

        # Test data loaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Get a batch to verify data format
        train_batch = next(iter(train_loader))
        assert len(train_batch) == 2  # images, masks
        images, masks = train_batch
        assert images.dim() == 4  # [B, C, H, W]
        assert masks.dim() == 4   # [B, C, H, W]

        # Create model
        model = SegmentationModel(config)

        # Verify model can process data
        with torch.no_grad():
            outputs = model(images)
            assert outputs.shape == masks.shape

        # Create trainer with minimal settings
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )

        # Train for one epoch
        trainer.fit(model, datamodule)

        # Verify training completed
        assert trainer.current_epoch == 0  # 0-indexed, so first epoch is 0

    def test_fast_dev_run(self, test_config_file):
        """Test fast dev run (1 batch) completes successfully."""
        config = load_config(test_config_file)

        # Create data module
        datamodule = SegmentationDataModule(config)
        datamodule.setup("fit")

        # Create model
        model = SegmentationModel(config)

        # Create trainer with fast dev run
        trainer = pl.Trainer(
            fast_dev_run=True,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )

        # Should complete without errors
        trainer.fit(model, datamodule)

    def test_validation_step(self, test_config_file):
        """Test that validation step works correctly."""
        config = load_config(test_config_file)

        # Create components
        datamodule = SegmentationDataModule(config)
        datamodule.setup("fit")
        model = SegmentationModel(config)

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False
        )

        # Run validation
        val_results = trainer.validate(model, datamodule)

        # Should return validation metrics
        assert len(val_results) > 0
        assert isinstance(val_results[0], dict)

    def test_model_predictions(self, test_config_file):
        """Test that model produces reasonable predictions."""
        config = load_config(test_config_file)

        # Create components
        datamodule = SegmentationDataModule(config)
        datamodule.setup("fit")
        model = SegmentationModel(config)

        # Get a batch of data
        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))
        images, masks = batch

        # Get predictions
        model.eval()
        with torch.no_grad():
            logits = model(images)
            predictions = torch.sigmoid(logits)

        # Verify prediction properties
        assert predictions.shape == masks.shape
        assert predictions.min() >= 0.0
        assert predictions.max() <= 1.0

    def test_metrics_computation(self, test_config_file):
        """Test that metrics are computed correctly during training."""
        config = load_config(test_config_file)

        # Create components
        datamodule = SegmentationDataModule(config)
        datamodule.setup("fit")
        model = SegmentationModel(config)

        # Get a validation batch
        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))

        # Manually run a validation step
        model.eval()
        result = model.validation_step(batch, 0)

        # Should return loss
        assert isinstance(result, torch.Tensor)
        assert result.dim() == 0  # Scalar loss

        # Check that metrics were updated
        for name, metric in model.val_metrics.items():
            # Metrics should have some state after update
            assert hasattr(metric, 'intersection') or hasattr(metric, 'jaccard')

    def test_different_architectures(self, test_config, temp_dir):
        """Test that different model architectures work."""
        architectures = ["Unet", "UnetPlusPlus"]

        for arch in architectures:
            # Create config for this architecture
            config = test_config.copy()
            config["model"]["architecture"] = arch
            config["output"]["dir"] = str(temp_dir / f"runs_{arch}")

            try:
                # Create components
                datamodule = SegmentationDataModule(config)
                datamodule.setup("fit")
                model = SegmentationModel(config)

                # Create trainer
                trainer = pl.Trainer(
                    fast_dev_run=True,
                    accelerator="cpu",
                    devices=1,
                    logger=False,
                    enable_checkpointing=False,
                    enable_progress_bar=False
                )

                # Should complete without errors
                trainer.fit(model, datamodule)

            except Exception as e:
                pytest.fail(f"Architecture {arch} failed: {e}")

    def test_different_loss_functions(self, test_config):
        """Test that different loss functions work."""
        loss_types = ["dice", "tversky", "bce"]

        for loss_type in loss_types:
            # Create config for this loss
            config = test_config.copy()
            config["training"]["loss"]["type"] = loss_type

            if loss_type == "tversky":
                config["training"]["loss"]["params"] = {"alpha": 0.5, "beta": 0.5}

            try:
                # Create model (this will create the loss function)
                model = SegmentationModel(config)

                # Test that loss function works
                import torch
                pred = torch.randn(2, 1, 32, 32)
                target = torch.randint(0, 2, (2, 1, 32, 32)).float()

                loss = model.loss_fn(pred, target)
                assert isinstance(loss, torch.Tensor)
                assert loss.dim() == 0

            except Exception as e:
                pytest.fail(f"Loss {loss_type} failed: {e}")


class TestDataModule:
    """Test DataModule functionality."""

    def test_datamodule_setup(self, test_config):
        """Test that DataModule setup works correctly."""
        datamodule = SegmentationDataModule(test_config)

        # Setup for fit
        datamodule.setup("fit")

        # Check that datasets are created
        assert datamodule.train_dataset is not None
        assert datamodule.val_dataset is not None

        # Check dataset sizes
        dataset_info = datamodule.get_dataset_info()
        assert dataset_info["train_size"] > 0
        assert dataset_info["val_size"] > 0
        assert dataset_info["train_size"] + dataset_info["val_size"] == 5  # Total dummy samples

    def test_dataloader_creation(self, test_config):
        """Test that dataloaders are created correctly."""
        datamodule = SegmentationDataModule(test_config)
        datamodule.setup("fit")

        # Get dataloaders
        train_loader = datamodule.train_dataloader()
        val_loader = datamodule.val_dataloader()

        # Check properties
        assert train_loader.batch_size == test_config["training"]["batch_size"]
        assert val_loader.batch_size == test_config["training"]["batch_size"]

        # Check that we can iterate
        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        assert len(train_batch) == 2
        assert len(val_batch) == 2

    def test_reproducible_splits(self, test_config):
        """Test that data splits are reproducible."""
        # Create two data modules with same config
        dm1 = SegmentationDataModule(test_config)
        dm2 = SegmentationDataModule(test_config)

        dm1.setup("fit")
        dm2.setup("fit")

        # Should have same splits
        info1 = dm1.get_dataset_info()
        info2 = dm2.get_dataset_info()

        assert info1["train_ids"] == info2["train_ids"]
        assert info1["val_ids"] == info2["val_ids"]


# Import torch here to avoid import errors in test discovery
import torch