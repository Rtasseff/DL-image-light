"""
Reproducibility utilities for deterministic training.

This module provides functions to ensure reproducible experiments
by setting random seeds and enabling deterministic behavior.
"""

import random
import os
import numpy as np
import torch
import pytorch_lightning as pl


def set_global_seed(seed: int = 42) -> None:
    """
    Set global random seed for reproducible experiments.

    Sets seeds for Python's random, NumPy, PyTorch, and PyTorch Lightning
    to ensure deterministic behavior across runs.

    Args:
        seed: Random seed value

    Example:
        >>> set_global_seed(42)
        # All random operations will now be deterministic
    """
    # Python random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch Lightning
    pl.seed_everything(seed, workers=True)

    # Additional deterministic settings
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for deterministic behavior
    os.environ['PYTHONHASHSEED'] = str(seed)


def enable_deterministic_mode() -> None:
    """
    Enable additional deterministic settings for PyTorch.

    This function enables deterministic algorithms in PyTorch
    for maximum reproducibility. Note that this may impact
    performance.
    """
    torch.use_deterministic_algorithms(True)

    # Handle non-deterministic operations
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'