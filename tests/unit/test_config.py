"""
Unit tests for configuration system.
"""

import pytest
import tempfile
from pathlib import Path
import yaml
from pydantic import ValidationError

from src.core.config import load_config, load_and_validate_config, Config


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config_loads_successfully(self):
        """Test that a valid configuration loads without errors."""
        config_data = {
            "project_name": "test_project",
            "task": "segmentation",
            "dataset": {
                "name": "test_dataset",
                "images_dir": "./data/images",
                "masks_dir": "./data/masks",
                "split": {
                    "type": "random",
                    "val_ratio": 0.2,
                    "seed": 42
                }
            },
            "model": {
                "architecture": "Unet",
                "encoder": "resnet34",
                "classes": 1
            },
            "training": {
                "epochs": 10,
                "batch_size": 4,
                "learning_rate": 1e-4,
                "loss": {
                    "type": "dice",
                    "params": {"smooth": 1.0}
                }
            }
        }

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Load and validate
            config = load_config(temp_path)
            assert config["project_name"] == "test_project"
            assert config["model"]["architecture"] == "Unet"
            assert config["training"]["epochs"] == 10
        finally:
            Path(temp_path).unlink()

    def test_invalid_task_raises_error(self):
        """Test that invalid task type raises validation error."""
        config_data = {
            "project_name": "test_project",
            "task": "invalid_task",  # Invalid task
            "dataset": {
                "name": "test_dataset",
                "images_dir": "./data/images",
                "masks_dir": "./data/masks"
            },
            "model": {
                "architecture": "Unet",
                "encoder": "resnet34"
            },
            "training": {
                "loss": {"type": "dice"}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_invalid_architecture_raises_error(self):
        """Test that invalid model architecture raises validation error."""
        config_data = {
            "project_name": "test_project",
            "task": "segmentation",
            "dataset": {
                "name": "test_dataset",
                "images_dir": "./data/images",
                "masks_dir": "./data/masks"
            },
            "model": {
                "architecture": "InvalidArch",  # Invalid architecture
                "encoder": "resnet34"
            },
            "training": {
                "loss": {"type": "dice"}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_missing_required_fields_raises_error(self):
        """Test that missing required fields raise validation error."""
        config_data = {
            "project_name": "test_project",
            # Missing task, dataset, model, training
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_nonexistent_file_raises_error(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_file.yaml")

    def test_default_values_applied(self):
        """Test that default values are applied correctly."""
        config_data = {
            "project_name": "test_project",
            "dataset": {
                "name": "test_dataset",
                "images_dir": "./data/images",
                "masks_dir": "./data/masks"
            },
            "model": {
                "architecture": "Unet",
                "encoder": "resnet34"
            },
            "training": {
                "loss": {"type": "dice"}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            # Check defaults
            assert config["task"] == "segmentation"
            assert config["model"]["in_channels"] == 3
            assert config["model"]["classes"] == 1
            assert config["training"]["batch_size"] == 8
            assert config["training"]["optimizer"] == "adamw"
            assert config["compute"]["accelerator"] == "auto"
        finally:
            Path(temp_path).unlink()

    def test_validation_ranges(self):
        """Test that validation ranges work correctly."""
        config_data = {
            "project_name": "test_project",
            "dataset": {
                "name": "test_dataset",
                "images_dir": "./data/images",
                "masks_dir": "./data/masks",
                "split": {
                    "val_ratio": 1.5  # Invalid: > 1.0
                }
            },
            "model": {
                "architecture": "Unet",
                "encoder": "resnet34"
            },
            "training": {
                "loss": {"type": "dice"}
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            with pytest.raises(ValidationError):
                load_config(temp_path)
        finally:
            Path(temp_path).unlink()