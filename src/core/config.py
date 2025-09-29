"""
Configuration validation and loading with Pydantic.

This module provides type-safe configuration loading and validation
using Pydantic models to ensure all required parameters are present
and correctly typed. Implements SDD v4.0 validation modes.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from enum import Enum
import yaml
from pydantic import BaseModel, Field, validator


class ValidationMode(Enum):
    """Configuration validation modes per SDD v4.0."""
    STRICT = "strict"      # Full validation, no missing fields
    PERMISSIVE = "permissive"  # Warnings on issues, fills defaults
    MINIMAL = "minimal"    # Only essential fields required


def get_validation_mode() -> ValidationMode:
    """
    Get validation mode based on environment.

    Returns:
        ValidationMode: Auto-detected or explicitly set mode
    """
    # Explicit override takes precedence
    if mode := os.environ.get('CONFIG_VALIDATION_MODE'):
        return ValidationMode(mode.lower())

    # CI environment detection
    if os.environ.get('CI'):
        if 'unit' in os.environ.get('TEST_SUITE', ''):
            return ValidationMode.MINIMAL
        return ValidationMode.STRICT

    # Local development detection
    if os.environ.get('USER'):  # Local machine
        return ValidationMode.PERMISSIVE

    # Default to STRICT for safety
    return ValidationMode.STRICT


class SplitConfig(BaseModel):
    """Configuration for data splitting."""
    type: str = Field(default="random", description="Split type: random, stratified, group")
    val_ratio: float = Field(default=0.2, ge=0.0, le=1.0, description="Validation split ratio")
    seed: int = Field(default=42, description="Random seed for splitting")


class DatasetConfig(BaseModel):
    """Configuration for dataset loading."""
    name: str = Field(description="Dataset name")
    images_dir: str = Field(description="Path to images directory")
    masks_dir: str = Field(description="Path to masks directory")
    image_suffix: str = Field(default=".png", description="Image file suffix")
    mask_suffix: str = Field(default="_mask.png", description="Mask file suffix")
    split: SplitConfig = Field(default_factory=SplitConfig, description="Split configuration")


class ModelConfig(BaseModel):
    """Configuration for model architecture."""
    architecture: str = Field(description="Model architecture: Unet, UnetPlusPlus, DeepLabV3Plus")
    encoder: str = Field(default="resnet34", description="Encoder backbone")
    encoder_weights: Optional[str] = Field(default="imagenet", description="Encoder pretrained weights")
    in_channels: int = Field(default=3, ge=1, description="Number of input channels")
    classes: int = Field(default=1, ge=1, description="Number of output classes")


class LossConfig(BaseModel):
    """Configuration for loss function."""
    type: str = Field(description="Loss type: dice, tversky, bce, focal")
    params: Dict[str, Any] = Field(default_factory=dict, description="Loss function parameters")


class SchedulerConfig(BaseModel):
    """Configuration for learning rate scheduler."""
    type: str = Field(description="Scheduler type: cosine, step, plateau")
    min_lr: Optional[float] = Field(default=1e-6, description="Minimum learning rate")
    step_size: Optional[int] = Field(default=10, description="Step size for StepLR")
    gamma: Optional[float] = Field(default=0.1, description="Gamma for StepLR")
    factor: Optional[float] = Field(default=0.5, description="Factor for ReduceLROnPlateau")
    patience: Optional[int] = Field(default=5, description="Patience for ReduceLROnPlateau")


class TrainingConfig(BaseModel):
    """Configuration for training parameters."""
    epochs: int = Field(default=50, ge=1, description="Number of training epochs")
    batch_size: int = Field(default=8, ge=1, description="Batch size")
    learning_rate: float = Field(default=1e-4, gt=0, description="Learning rate")
    optimizer: str = Field(default="adamw", description="Optimizer: adam, adamw, sgd")
    weight_decay: float = Field(default=1e-4, ge=0, description="Weight decay")
    scheduler: Optional[SchedulerConfig] = Field(default=None, description="Scheduler configuration")
    loss: LossConfig = Field(description="Loss configuration")
    metrics: List[str] = Field(default=["dice"], description="Metrics to track")
    accumulate_grad_batches: int = Field(default=1, ge=1, description="Gradient accumulation batches")


class AugmentationConfig(BaseModel):
    """Configuration for data augmentations."""
    name: str = Field(description="Augmentation name")
    params: Dict[str, Any] = Field(default_factory=dict, description="Augmentation parameters")


class AugmentationsConfig(BaseModel):
    """Configuration for train/val augmentations."""
    train: List[AugmentationConfig] = Field(default_factory=list, description="Training augmentations")
    val: List[AugmentationConfig] = Field(default_factory=list, description="Validation augmentations")


class CheckpointConfig(BaseModel):
    """Configuration for model checkpointing."""
    monitor: str = Field(default="val_loss", description="Metric to monitor for checkpointing")
    mode: str = Field(default="min", description="Mode: min or max")
    save_best_only: bool = Field(default=True, description="Save only best checkpoint")


class OutputConfig(BaseModel):
    """Configuration for output settings."""
    dir: str = Field(default="./runs", description="Output directory")
    save_overlays: bool = Field(default=True, description="Save overlay visualizations")
    save_predictions: bool = Field(default=True, description="Save prediction arrays")
    checkpoint: CheckpointConfig = Field(default_factory=CheckpointConfig, description="Checkpoint settings")


class ComputeConfig(BaseModel):
    """Configuration for compute settings."""
    accelerator: str = Field(default="auto", description="Accelerator: cpu, gpu, mps, auto")
    devices: int = Field(default=1, ge=1, description="Number of devices")
    precision: int = Field(default=32, description="Training precision: 16, 32")
    deterministic: bool = Field(default=True, description="Enable deterministic training")
    seed: int = Field(default=42, description="Global random seed")


class LoggingConfig(BaseModel):
    """Configuration for logging."""
    level: str = Field(default="INFO", description="Logging level")
    save_tensorboard: bool = Field(default=False, description="Save TensorBoard logs")


class ResourcesConfig(BaseModel):
    """Configuration for resource management (SDD v4.0)."""
    auto_tune: bool = Field(default=False, description="Enable auto-tuning")
    log_effective_settings: bool = Field(default=True, description="Log effective settings")


class VisualizationConfig(BaseModel):
    """Configuration for visualization (SDD v4.0)."""
    mode: str = Field(default="simple", description="Visualization mode: simple, detailed")
    overlay_alpha: float = Field(default=0.3, ge=0.0, le=1.0, description="Overlay transparency")
    overlay_colormap: str = Field(default="viridis", description="Overlay colormap")


class Config(BaseModel):
    """Main configuration class with SDD v4.0 support."""
    project_name: str = Field(description="Project name")
    task: str = Field(default="segmentation", description="Task type: segmentation")

    # SDD v4.0 specific fields
    use_fallbacks: Optional[bool] = Field(default=False, description="Use fallback implementations")
    validation_mode: Optional[str] = Field(default=None, description="Config validation mode")

    dataset: DatasetConfig = Field(description="Dataset configuration")
    model: ModelConfig = Field(description="Model configuration")
    training: TrainingConfig = Field(description="Training configuration")
    augmentations: AugmentationsConfig = Field(default_factory=AugmentationsConfig, description="Augmentations")
    output: OutputConfig = Field(default_factory=OutputConfig, description="Output configuration")
    compute: ComputeConfig = Field(default_factory=ComputeConfig, description="Compute configuration")
    logging: LoggingConfig = Field(default_factory=LoggingConfig, description="Logging configuration")

    # SDD v4.0 specific configs
    resources: ResourcesConfig = Field(default_factory=ResourcesConfig, description="Resource management")
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig, description="Visualization settings")

    @validator("task")
    def validate_task(cls, v):
        """Validate task type."""
        allowed_tasks = ["segmentation"]
        if v not in allowed_tasks:
            raise ValueError(f"Task must be one of {allowed_tasks}, got {v}")
        return v

    @validator("model")
    def validate_model_architecture(cls, v):
        """Validate model architecture."""
        allowed_archs = ["Unet", "UnetPlusPlus", "DeepLabV3Plus"]
        if v.architecture not in allowed_archs:
            raise ValueError(f"Architecture must be one of {allowed_archs}, got {v.architecture}")
        return v


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValidationError: If config validation fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    # Validate configuration
    config = Config(**config_dict)

    # Return as dictionary for easier access
    return config.model_dump()


def load_and_validate_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and validate configuration with enhanced error reporting.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Validated configuration dictionary
    """
    try:
        return load_config(config_path)
    except Exception as e:
        print(f"Error loading configuration from {config_path}:")
        print(f"  {type(e).__name__}: {e}")
        raise