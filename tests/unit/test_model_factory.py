"""Unit tests for the model factory."""

import pytest

pytest.importorskip("pytorch_lightning")
pytest.importorskip("segmentation_models_pytorch")

import torch.nn as nn

from src.models.factory import build_model


def _base_config() -> dict:
    return {
        "model": {
            "architecture": "unet",
            "encoder": "resnet34",
            "in_channels": 3,
            "out_channels": 1,
        }
    }


def test_build_model_returns_module():
    config = _base_config()
    model = build_model(config)
    assert isinstance(model, nn.Module)


def test_unknown_architecture_raises_value_error():
    config = _base_config()
    config["model"]["architecture"] = "unknown"
    with pytest.raises(ValueError):
        build_model(config)
