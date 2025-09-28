"""
Pytest fixtures for testing.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import yaml
import torch


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def dummy_dataset(temp_dir):
    """Create dummy dataset for testing."""
    # Create directories
    images_dir = temp_dir / "images"
    masks_dir = temp_dir / "masks"
    images_dir.mkdir()
    masks_dir.mkdir()

    # Create dummy images and masks
    for i in range(5):
        # Create RGB image
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(image).save(images_dir / f"image_{i:03d}.png")

        # Create binary mask
        mask = np.random.randint(0, 2, (64, 64), dtype=np.uint8) * 255
        Image.fromarray(mask, mode='L').save(masks_dir / f"image_{i:03d}_mask.png")

    return {
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "num_samples": 5
    }


@pytest.fixture
def test_config(dummy_dataset):
    """Create test configuration."""
    return {
        "project_name": "test_project",
        "task": "segmentation",
        "dataset": {
            "name": "test_dataset",
            "images_dir": dummy_dataset["images_dir"],
            "masks_dir": dummy_dataset["masks_dir"],
            "image_suffix": ".png",
            "mask_suffix": "_mask.png",
            "split": {
                "type": "random",
                "val_ratio": 0.2,
                "seed": 42
            }
        },
        "model": {
            "architecture": "Unet",
            "encoder": "resnet34",
            "encoder_weights": "imagenet",
            "in_channels": 3,
            "classes": 1
        },
        "training": {
            "epochs": 2,
            "batch_size": 2,
            "learning_rate": 1e-4,
            "optimizer": "adamw",
            "weight_decay": 1e-4,
            "loss": {
                "type": "dice",
                "params": {"smooth": 1.0}
            },
            "metrics": ["dice", "iou"],
            "accumulate_grad_batches": 1
        },
        "augmentations": {
            "train": [
                {
                    "name": "Resize",
                    "params": {"height": 64, "width": 64}
                }
            ],
            "val": [
                {
                    "name": "Resize",
                    "params": {"height": 64, "width": 64}
                }
            ]
        },
        "output": {
            "dir": "./test_runs",
            "save_overlays": False,
            "save_predictions": False,
            "checkpoint": {
                "monitor": "val/loss",
                "mode": "min"
            }
        },
        "compute": {
            "accelerator": "cpu",  # Use CPU for tests
            "devices": 1,
            "precision": 32,
            "deterministic": True,
            "seed": 42
        },
        "logging": {
            "level": "WARNING"  # Reduce log noise in tests
        }
    }


@pytest.fixture
def test_config_file(test_config, temp_dir):
    """Create test configuration file."""
    config_file = temp_dir / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(test_config, f)
    return str(config_file)


@pytest.fixture
def sample_tensors():
    """Create sample tensors for testing."""
    torch.manual_seed(42)
    return {
        "logits": torch.randn(2, 1, 32, 32),
        "probs": torch.sigmoid(torch.randn(2, 1, 32, 32)),
        "binary_mask": torch.randint(0, 2, (2, 1, 32, 32)).float(),
        "batch_size": 2,
        "height": 32,
        "width": 32
    }


@pytest.fixture
def mock_model_output():
    """Create mock model output for testing."""
    torch.manual_seed(42)
    batch_size = 2
    height, width = 64, 64

    return {
        "logits": torch.randn(batch_size, 1, height, width),
        "probabilities": torch.sigmoid(torch.randn(batch_size, 1, height, width)),
        "predictions": torch.randint(0, 2, (batch_size, 1, height, width)).float()
    }