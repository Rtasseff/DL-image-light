"""Utilities for capturing runtime environment information."""

from typing import Any, Dict


def collect_environment_info() -> Dict[str, Any]:
    """Return a dictionary describing the ML environment."""
    import sys
    import torch
    import torchvision
    import pytorch_lightning as pl
    import segmentation_models_pytorch as smp
    import albumentations as A

    return {
        "python": sys.version,
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "lightning": pl.__version__,
        "smp": smp.__version__,
        "albumentations": A.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "mps_available": torch.backends.mps.is_available() if hasattr(torch.backends, "mps") else False,
    }
