# Image Segmentation Platform — Software Design Document
*Version:* 3.0  
*Date:* 2025-01-15  
*Status:* Living Document - Updated with Implementation Lessons

## Executive Summary
Build a **production-ready** segmentation platform using **proven libraries** with **robust fallback strategies** for dependency management and testing. Based on real implementation experience through Week 2 MVP.

**Key Updates in v3.0:**
- Comprehensive dependency management with fallback implementations
- Three-tier testing strategy for development without heavy dependencies  
- Detailed visualization specifications
- Resource management and memory limits
- Refined timeline based on actual implementation
- Implementation guidance from lessons learned

## 1. Purpose & Scope

### Current Scope (MVP - Completed Week 2)
- **Task:** Semantic segmentation
- **Models:** U-Net/U-Net++/DeepLabV3+ via SMP with fallbacks
- **Data:** Local PNG/JPEG images with mask labels
- **Validation:** Single train/val split with fixed seed
- **Target:** DRIVE dataset (retinal vessel segmentation)
- **Platform:** M1 MacBook (CPU/MPS), Linux + CUDA ready

### Immediate Next Phase (v3.1)
- **5-fold cross-validation** with robust error handling
- **Test-time augmentation** with memory management
- **Sliding window inference** for large images
- **Enhanced reporting** with interactive HTML

### Future Scope (v4+)
- Classification tasks with same infrastructure
- MONAI integration for medical imaging
- Multimodal (image + tabular) fusion

## 2. Core Technical Stack

### Dependency Strategy

```yaml
dependencies:
  # Core Minimal (Always Required)
  core:
    - torch: ">=2.0.0,<3.0.0"
    - numpy: ">=1.20.0"
    - pyyaml: ">=6.0"
    - pydantic: ">=2.0.0"
    - pillow: ">=9.0.0"
    - tqdm: ">=4.60.0"
    
  # ML Performance (Full Functionality)
  ml_required:
    - pytorch-lightning: ">=2.0.0,<3.0.0"
    - segmentation-models-pytorch: ">=0.3.0"
    
  ml_optional:
    - torchmetrics: ">=1.0.0"  # Fallback: simple numpy metrics
    - albumentations: ">=1.3.0"  # Fallback: torchvision transforms
    - timm: ">=0.9.0"  # Fallback: torchvision models
    
  # Visualization (Reporting)
  visualization:
    - matplotlib: ">=3.5.0"
    - seaborn: ">=0.12.0"  # Optional: enhanced plots
    - jinja2: ">=3.0.0"  # Optional: HTML reports
    
  # Development (Testing/Linting)
  development:
    - pytest: ">=7.0.0"
    - pytest-cov: ">=4.0.0"
    - black: ">=23.0.0"
    - isort: ">=5.0.0"
    - mypy: ">=1.0.0"
```

### Fallback Implementation Strategy

```python
# Required structure for all optional dependencies
# src/core/dependencies.py

import importlib.util
import warnings
from typing import Dict, Any

class DependencyManager:
    """Manages optional dependencies with fallbacks."""
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check which optional dependencies are available."""
        deps = {}
        deps['torchmetrics'] = importlib.util.find_spec("torchmetrics") is not None
        deps['albumentations'] = importlib.util.find_spec("albumentations") is not None
        deps['timm'] = importlib.util.find_spec("timm") is not None
        deps['seaborn'] = importlib.util.find_spec("seaborn") is not None
        return deps
    
    @staticmethod
    def warn_missing(package: str, fallback: str):
        """Issue warning about missing package."""
        warnings.warn(
            f"{package} not available, using {fallback}. "
            f"Install with: pip install {package}",
            UserWarning
        )

# Example usage in metrics module
# src/metrics/__init__.py

from .dependencies import DependencyManager

dm = DependencyManager()
deps = dm.check_dependencies()

if deps['torchmetrics']:
    from torchmetrics import Dice, JaccardIndex, Precision, Recall
else:
    dm.warn_missing('torchmetrics', 'simple numpy implementations')
    from .simple import SimpleDice as Dice
    from .simple import SimpleIoU as JaccardIndex
    from .simple import SimplePrecision as Precision
    from .simple import SimpleRecall as Recall
```

## 3. Architecture: Five Stable Interfaces

### API Stability Contract
These interfaces remain stable through v4.0:

### 3.1 DataModule Contract
```python
from typing import Protocol, Optional
from torch.utils.data import DataLoader

class DataModule(Protocol):
    """
    Stable API v3.0 - v4.0
    Owns data loading, splits, transforms, and memory management.
    """
    
    def setup(self, stage: str) -> None: 
        """Setup data for train/val/test."""
        ...
    
    def train_dataloader(self) -> DataLoader: 
        """Return training dataloader with auto batch size."""
        ...
    
    def val_dataloader(self) -> DataLoader: 
        """Return validation dataloader."""
        ...
    
    def test_dataloader(self) -> Optional[DataLoader]: 
        """Return test dataloader if available."""
        ...
    
    def auto_adjust_batch_size(self, available_memory_mb: int) -> None:
        """Adjust batch size based on available memory."""
        ...
```

### 3.2 Model Contract
```python
class Model(nn.Module):
    """
    Stable API v3.0 - v4.0
    Pure PyTorch model with optional prediction strategies.
    """
    
    def forward(self, x: Tensor) -> Tensor: 
        """Standard forward pass."""
        ...
    
    def predict_step(self, x: Tensor, strategy: str = "standard") -> Tensor:
        """
        Prediction with different strategies.
        Args:
            x: Input tensor
            strategy: "standard", "tta", "sliding_window"
        """
        if strategy == "standard":
            return self.forward(x)
        elif strategy == "tta":
            return self._predict_with_tta(x)
        elif strategy == "sliding_window":
            return self._predict_sliding_window(x)
```

### 3.3 Loss Contract
```python
class Loss(Protocol):
    """Stable API v3.0 - v4.0"""
    
    def __call__(
        self, 
        pred: Tensor, 
        target: Tensor, 
        **kwargs: Any
    ) -> Tensor: 
        """Compute loss with optional sample weights."""
        ...
```

### 3.4 Metrics Contract
```python
class Metrics(Protocol):
    """Stable API v3.0 - v4.0"""
    
    def update(self, pred: Tensor, target: Tensor) -> None: 
        """Update metric state."""
        ...
    
    def compute(self) -> Dict[str, Tensor]: 
        """Compute final metric values."""
        ...
    
    def reset(self) -> None: 
        """Reset metric state."""
        ...
```

### 3.5 Trainer Contract
```python
class Trainer(Protocol):
    """Stable API v3.0 - v4.0"""
    
    def fit(
        self, 
        model: Model, 
        datamodule: DataModule, 
        callbacks: Optional[List] = None
    ) -> None: 
        """Train model."""
        ...
    
    def validate(
        self, 
        model: Model, 
        datamodule: DataModule
    ) -> List[Dict]: 
        """Validate model."""
        ...
    
    def test(
        self, 
        model: Model, 
        datamodule: DataModule
    ) -> List[Dict]: 
        """Test model."""
        ...
```

## 4. Configuration Schema with Validation

### Configuration Validation Strategy

```python
# src/core/config.py

from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, validator
import warnings

class ValidationMode(str, Enum):
    """Configuration validation modes."""
    STRICT = "strict"        # Production - all fields required
    PERMISSIVE = "permissive"  # Development - allows extras
    MINIMAL = "minimal"      # Testing - only essentials

class ConfigValidator:
    """Multi-mode configuration validation."""
    
    @classmethod
    def validate(
        cls, 
        config: Dict[str, Any], 
        mode: ValidationMode = ValidationMode.STRICT
    ) -> Dict[str, Any]:
        """
        Validate configuration based on mode.
        
        Args:
            config: Raw configuration dictionary
            mode: Validation strictness level
            
        Returns:
            Validated configuration
            
        Raises:
            ValidationError: In STRICT mode if config invalid
        """
        if mode == ValidationMode.MINIMAL:
            # Only validate essential fields
            required_fields = [
                "model.architecture",
                "dataset.images_dir",
                "dataset.masks_dir"
            ]
            for field in required_fields:
                if not cls._has_nested_key(config, field):
                    raise ValueError(f"Missing required field: {field}")
                    
        elif mode == ValidationMode.PERMISSIVE:
            # Validate known fields, warn on unknown
            validated = cls._validate_schema(config, warn_extra=True)
            return validated
            
        else:  # STRICT
            # Full validation, no unknown fields
            validated = cls._validate_schema(config, warn_extra=False)
            return validated
        
        return config
```

### Configuration Schema
```yaml
# config.yaml with validation annotations

# Project metadata
project_name: "drive_vessel_segmentation"  # Required in STRICT
task: "segmentation"  # Required, enum: ["segmentation", "classification"]
validation_mode: "permissive"  # For development

# Dataset configuration
dataset:
  name: "DRIVE"  # Required
  images_dir: "./data/drive/images"  # Required
  masks_dir: "./data/drive/masks"  # Required for segmentation
  
  # Optional with defaults
  image_suffix: ".png"  # Default: ".png"
  mask_suffix: "_mask.png"  # Default: "_mask.png"
  
  # Memory management
  cache_in_memory: false  # Default: false
  max_cache_size_mb: 1000  # Default: 1000
  
  # Train/val split
  split:
    type: "random"  # enum: ["random", "stratified", "group"]
    val_ratio: 0.2  # Required if type="random"
    seed: 42  # Required

# Model configuration  
model:
  architecture: "Unet"  # Required, enum: ["Unet", "UnetPlusPlus", "DeepLabV3Plus"]
  encoder: "resnet34"  # Required
  encoder_weights: "imagenet"  # Optional, default: "imagenet"
  
  # Architecture specific
  in_channels: 3  # Default: 3
  classes: 1  # Default: 1 for binary
  
  # Fallback for missing SMP
  use_simple_unet: false  # Use basic implementation if SMP unavailable

# Training configuration
training:
  epochs: 50  # Required
  batch_size: 8  # Required, auto-adjusted based on memory
  learning_rate: 1e-4  # Required
  
  # Optional with defaults
  optimizer: "adamw"  # Default: "adamw"
  weight_decay: 1e-4  # Default: 1e-4
  
  # Early stopping
  early_stopping:
    enabled: true  # Default: true
    monitor: "val_loss"  # Default: "val_loss"
    patience: 10  # Default: 10
    mode: "min"  # Default: "min"
  
  # Loss function
  loss:
    type: "dice"  # Required, enum: ["dice", "tversky", "bce", "focal", "compound"]
    params:  # Type-specific parameters
      smooth: 1.0
      # For tversky: alpha, beta
      # For focal: alpha, gamma
  
  # Metrics to track
  metrics:  # Default: ["dice"]
    - "dice"
    - "iou"
    - "precision"
    - "recall"

# Resource management
resources:
  memory:
    max_batch_size_4gb: 4  # Auto batch size limits
    max_batch_size_8gb: 8
    max_batch_size_16gb: 16
    warn_threshold: 0.8  # Warn at 80% memory usage
    fail_threshold: 0.95  # Fail at 95% memory usage
    
  inference:
    tile_size: [512, 512]  # For large images
    tile_overlap: 64  # Pixels
    batch_processing: true
    max_batch_size: 4  # For inference

# Augmentations
augmentations:
  backend: "auto"  # "albumentations", "torchvision", "auto"
  train:
    - name: "RandomRotate90"
      params: {p: 0.5}
    - name: "HorizontalFlip"
      params: {p: 0.5}
  val: []  # No augmentation for validation

# Visualization specifications
visualization:
  overlays:
    enabled: true
    colormap: "viridis"  # matplotlib colormap
    alpha: 0.3  # Transparency
    dpi: 150  # Output resolution
    formats: ["png"]  # Output formats
    
  error_analysis:
    enabled: false  # Set true for detailed error maps
    colors:
      true_positive: [0, 255, 0, 128]  # Green
      false_positive: [255, 0, 0, 128]  # Red  
      false_negative: [0, 0, 255, 128]  # Blue
      
  metrics_plots:
    enabled: true
    figure_size: [10, 6]
    style: "seaborn"  # If available
    save_format: "svg"

# Output configuration
output:
  dir: "./runs"  # Required
  experiment_name: null  # Auto-generated if null
  save_predictions: false  # Save raw predictions (disk intensive)
  save_overlays: true
  checkpoint:
    monitor: "val_loss"
    mode: "min"
    save_best_only: true
    save_last: true  # Always keep last checkpoint

# Compute settings
compute:
  accelerator: "auto"  # "cpu", "cuda", "mps", "auto"
  devices: 1
  precision: 32  # 16 for mixed precision (if supported)
  deterministic: true  # Reproducibility
  seed: 42
  
  # Platform specific
  cuda:
    cudnn_benchmark: false  # For reproducibility
    amp_backend: "native"
  mps:
    fallback_to_cpu: true  # If operation unsupported

# Logging
logging:
  level: "INFO"  # "DEBUG", "INFO", "WARNING", "ERROR"
  console: true
  file: true
  tensorboard: false  # Requires tensorboard
  
  # Detailed logging
  log_gradients: false
  log_memory: true
  log_timing: true
```

## 5. Repository Structure

```
segmentation-platform/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Config validation with modes
│   │   ├── dependencies.py     # Dependency checking & fallbacks
│   │   ├── trainer.py          # Lightning trainer wrapper
│   │   ├── registry.py         # Component registry
│   │   └── errors.py           # Custom exceptions with helpful messages
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datamodule.py       # DataModule with auto batch sizing
│   │   ├── dataset.py          # Dataset with caching options
│   │   ├── transforms.py       # Transform builders with fallbacks
│   │   ├── memory.py           # Memory management utilities
│   │   └── readers/            # Extensible readers
│   │       ├── base.py
│   │       └── image.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── segmentation.py     # SMP wrapper with fallback
│   │   ├── simple_unet.py      # Fallback UNet implementation
│   │   ├── lightning_module.py # LightningModule
│   │   └── architectures/      # Future custom architectures
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── factory.py          # Loss factory with validation
│   │   ├── dice.py             # Dice variants
│   │   ├── tversky.py          # Tversky loss
│   │   ├── focal.py            # Focal loss
│   │   └── compound.py         # Combined losses
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── factory.py          # Metric factory
│   │   ├── torchmetrics_impl.py # TorchMetrics wrappers
│   │   └── simple.py           # Numpy fallback implementations
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict.py          # Standard prediction
│   │   ├── tta.py              # Test-time augmentation
│   │   ├── sliding_window.py   # Sliding window for large images
│   │   └── memory_manager.py   # Memory-aware inference
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── overlays.py         # Configurable overlays
│   │   ├── error_maps.py       # Error analysis visualizations
│   │   ├── metrics_plot.py     # Training curves with fallbacks
│   │   └── html_report.py      # Interactive HTML reports
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── io.py               # Robust file I/O
│   │   ├── logging.py          # Multi-level logging
│   │   ├── reproducibility.py  # Seed management
│   │   ├── memory.py           # Memory monitoring
│   │   └── platform.py         # Platform detection & optimization
│   │
│   └── testing/
│       ├── __init__.py
│       └── fixtures.py         # Shared test fixtures
│
├── configs/
│   ├── base_config.yaml        # Default configuration
│   ├── drive.yaml              # DRIVE dataset specific
│   ├── test_minimal.yaml       # Minimal config for testing
│   └── examples/
│       ├── low_memory.yaml    # Settings for 4GB systems
│       ├── high_quality.yaml  # Maximum quality settings
│       └── fast_training.yaml # Speed optimized settings
│
├── scripts/
│   ├── train.py                # Main training script with error handling
│   ├── predict.py              # Inference with multiple strategies
│   ├── evaluate.py             # Comprehensive evaluation
│   ├── visualize_results.py    # Generate all visualizations
│   └── check_dependencies.py   # Dependency diagnostic tool
│
├── tests/
│   ├── unit/                   # No ML dependencies required
│   │   ├── test_config.py
│   │   ├── test_io.py
│   │   └── test_registry.py
│   │
│   ├── integration/
│   │   ├── minimal/           # Core dependencies only
│   │   │   ├── test_pipeline.py
│   │   │   └── test_data_loading.py
│   │   │
│   │   └── full/              # All dependencies required
│   │       ├── test_training.py
│   │       ├── test_augmentations.py
│   │       └── test_metrics.py
│   │
│   └── fixtures/              # Test data and mocks
│       ├── tiny_dataset/
│       └── mock_models.py
│
├── runs/                      # Experiment outputs (gitignored)
│
├── docs/
│   ├── README.md              # Comprehensive documentation
│   ├── QUICKSTART.md          # 5-minute getting started
│   ├── CONFIG_GUIDE.md        # All configuration options
│   ├── ARCHITECTURE.md        # Technical architecture
│   ├── EXTENDING.md           # How to add components
│   ├── TROUBLESHOOTING.md     # Common issues and solutions
│   └── IMPLEMENTATION_LESSONS.md  # Living document of lessons learned
│
├── requirements/
│   ├── base.txt              # Core dependencies only
│   ├── ml.txt                # ML dependencies (includes base)
│   ├── viz.txt               # Visualization (includes base)
│   ├── dev.txt               # Development tools
│   └── all.txt               # Everything
│
├── setup.py                   # Package setup
├── pyproject.toml            # Modern Python project config
├── .gitignore
└── Makefile                  # Common commands
```

## 6. Testing Philosophy

### Three-Tier Testing Strategy

```python
# tests/conftest.py
"""Pytest configuration with dependency detection."""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.dependencies import DependencyManager

# Detect available dependencies
deps = DependencyManager().check_dependencies()

# Markers for different test levels
pytest.mark.unit = pytest.mark.unit  # No external deps
pytest.mark.integration_minimal = pytest.mark.integration_minimal  # Core deps
pytest.mark.integration_full = pytest.mark.integration_full  # All deps

# Skip decorators based on dependencies
requires_torchmetrics = pytest.mark.skipif(
    not deps['torchmetrics'],
    reason="TorchMetrics not installed"
)

requires_albumentations = pytest.mark.skipif(
    not deps['albumentations'],
    reason="Albumentations not installed"
)

# Auto-skip full integration tests if missing deps
def pytest_collection_modifyitems(config, items):
    """Auto-skip tests based on available dependencies."""
    for item in items:
        if "integration_full" in item.keywords:
            if not all(deps.values()):
                item.add_marker(pytest.mark.skip(
                    reason="Full dependencies not available"
                ))
```

### Test Categories

#### 1. Unit Tests (No ML Dependencies)
```python
# tests/unit/test_config.py
"""Test configuration without ML dependencies."""

def test_config_validation_minimal():
    """Test minimal config validation."""
    config = {
        "model": {"architecture": "Unet"},
        "dataset": {
            "images_dir": "/path/to/images",
            "masks_dir": "/path/to/masks"
        }
    }
    validated = ConfigValidator.validate(
        config, 
        mode=ValidationMode.MINIMAL
    )
    assert validated is not None

def test_config_missing_required():
    """Test that missing required fields raise errors."""
    config = {"model": {}}  # Missing architecture
    with pytest.raises(ValueError, match="Missing required field"):
        ConfigValidator.validate(config, mode=ValidationMode.MINIMAL)
```

#### 2. Integration Tests - Minimal
```python
# tests/integration/minimal/test_pipeline.py
"""Test pipeline with fallback implementations."""

@pytest.mark.integration_minimal
def test_pipeline_with_simple_metrics():
    """Test training with simple metric implementations."""
    # Uses simple numpy metrics instead of TorchMetrics
    from metrics.simple import SimpleDice
    
    metric = SimpleDice()
    pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()
    
    metric.update(pred, target)
    result = metric.compute()
    
    assert 0 <= result <= 1
```

#### 3. Integration Tests - Full
```python
# tests/integration/full/test_training.py
"""Test with all dependencies."""

@pytest.mark.integration_full
@requires_torchmetrics
def test_full_training_pipeline():
    """Test complete training with all features."""
    config = load_test_config("configs/test_minimal.yaml")
    trainer = Trainer(fast_dev_run=True)
    model = create_model(config)
    datamodule = create_datamodule(config)
    
    trainer.fit(model, datamodule)
    
    assert Path("runs").exists()
```

### Smoke Test
```python
# tests/test_smoke.py
"""Quick validation that everything works."""

@pytest.mark.smoke
def test_smoke(tmp_path):
    """1 epoch, 2 samples, all components touched."""
    config = {
        "model": {"architecture": "Unet", "encoder": "resnet18"},
        "dataset": {
            "images_dir": str(tmp_path / "images"),
            "masks_dir": str(tmp_path / "masks")
        },
        "training": {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001
        }
    }
    
    # Create tiny fake dataset
    create_tiny_dataset(tmp_path)
    
    # Should complete in <60 seconds
    start = time.time()
    run_training(config)
    assert time.time() - start < 60
```

## 7. Implementation Timeline (Revised)

### Week 1: Core Pipeline ✅ (Completed)
- DataModule with basic data loading
- SMP Model wrapper with fallbacks
- Basic training script
- Single train/val split
- Dice loss and metric

### Week 2A: Core Components ✅ (Completed)
- All losses (Dice, Tversky, BCE, Focal)
- All metrics with fallbacks
- Basic configuration system
- Simple overlay visualization

### Week 2B: Integration ✅ (Completed)
- Prediction script
- Evaluation pipeline
- Results aggregation
- Basic reporting

### Week 3: Polish & Robustness (Current)
- Comprehensive error handling
- Memory management
- Dependency diagnostics
- HTML reporting
- Complete documentation
- Three-tier testing

### Week 4: K-Fold CV (Next)
- Cross-validation orchestrator
- Fold aggregation
- Statistical analysis
- Enhanced reporting

## 8. Resource Management

### Memory Management Strategy

```python
# src/utils/memory.py
"""Memory management utilities."""

import torch
import psutil
import warnings
from typing import Tuple, Optional

class MemoryManager:
    """Manages memory for training and inference."""
    
    @staticmethod
    def get_available_memory() -> Tuple[int, int]:
        """
        Get available system and GPU memory in MB.
        
        Returns:
            (system_memory_mb, gpu_memory_mb)
        """
        # System memory
        mem = psutil.virtual_memory()
        system_mb = mem.available // (1024 * 1024)
        
        # GPU memory
        gpu_mb = 0
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.get_device_properties(0).total_memory
            gpu_mb = gpu_mb // (1024 * 1024)
        elif torch.backends.mps.is_available():
            # MPS doesn't provide memory info, use heuristic
            gpu_mb = 8000  # Assume 8GB for M1
            
        return system_mb, gpu_mb
    
    @staticmethod
    def suggest_batch_size(
        model_size: str = "medium",
        image_size: Tuple[int, int] = (512, 512),
        available_memory_mb: Optional[int] = None
    ) -> int:
        """
        Suggest batch size based on available memory.
        
        Args:
            model_size: "small", "medium", "large"
            image_size: (height, width) of images
            available_memory_mb: Override auto-detection
            
        Returns:
            Suggested batch size
        """
        if available_memory_mb is None:
            system_mb, gpu_mb = MemoryManager.get_available_memory()
            available_memory_mb = gpu_mb if gpu_mb > 0 else system_mb
        
        # Memory usage estimation (rough)
        pixels = image_size[0] * image_size[1]
        
        if model_size == "small":
            bytes_per_sample = pixels * 4 * 10  # ~10 layers
        elif model_size == "medium":
            bytes_per_sample = pixels * 4 * 20  # ~20 layers
        else:  # large
            bytes_per_sample = pixels * 4 * 40  # ~40 layers
            
        mb_per_sample = bytes_per_sample / (1024 * 1024)
        
        # Use 60% of available memory for safety
        safe_memory = available_memory_mb * 0.6
        batch_size = int(safe_memory / mb_per_sample)
        
        # Clamp to reasonable range
        batch_size = max(1, min(batch_size, 32))
        
        if batch_size < 4:
            warnings.warn(
                f"Low memory: batch size {batch_size}. "
                "Consider reducing image size or using gradient accumulation."
            )
        
        return batch_size
```

### Resource Limits Configuration

```yaml
# Resource limits in config
resources:
  memory:
    # Batch size limits by available memory
    batch_size_limits:
      2048: 2   # 2GB: batch_size=2
      4096: 4   # 4GB: batch_size=4
      8192: 8   # 8GB: batch_size=8
      16384: 16 # 16GB: batch_size=16
      32768: 32 # 32GB: batch_size=32
    
    # Monitoring thresholds
    monitor:
      check_interval: 10  # Check every N batches
      warn_threshold: 0.8  # Warn at 80% usage
      fail_threshold: 0.95 # Stop at 95% usage
      reduce_batch_on_warn: true
    
  inference:
    # Sliding window for large images
    sliding_window:
      enabled: "auto"  # auto, true, false
      window_size: [512, 512]
      overlap: 64
      batch_size: 4
    
    # Test-time augmentation
    tta:
      enabled: false
      transforms: ["hflip"]  # Start simple
      reduction: "mean"  # mean, max, gmean
```

## 9. Visualization Specifications

### Detailed Visualization Requirements

```python
# src/visualization/specs.py
"""Visualization specifications and defaults."""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt

@dataclass
class OverlaySpec:
    """Specifications for segmentation overlays."""
    colormap: str = "viridis"
    alpha: float = 0.3
    dpi: int = 150
    formats: List[str] = None
    error_colors: dict = None
    
    def __post_init__(self):
        if self.formats is None:
            self.formats = ["png"]
        if self.error_colors is None:
            self.error_colors = {
                'true_positive': [0, 255, 0, 128],   # Green
                'false_positive': [255, 0, 0, 128],  # Red
                'false_negative': [0, 0, 255, 128],  # Blue
                'true_negative': [0, 0, 0, 0]        # Transparent
            }
    
    def validate(self):
        """Validate specifications."""
        assert 0 <= self.alpha <= 1, "Alpha must be between 0 and 1"
        assert self.dpi > 0, "DPI must be positive"
        assert self.colormap in plt.colormaps(), f"Unknown colormap: {self.colormap}"

@dataclass
class PlotSpec:
    """Specifications for metric plots."""
    figure_size: Tuple[int, int] = (10, 6)
    style: str = "seaborn"
    save_format: str = "svg"
    show_grid: bool = True
    show_legend: bool = True
    
    def apply_style(self):
        """Apply matplotlib style with fallback."""
        try:
            plt.style.use(self.style)
        except:
            # Fallback to default if style not available
            plt.style.use('default')

# Default specifications
DEFAULT_OVERLAY_SPEC = OverlaySpec()
DEFAULT_PLOT_SPEC = PlotSpec()
```

### Visualization Implementation with Fallbacks

```python
# src/visualization/overlays.py
"""Overlay generation with configurable options."""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional

from .specs import OverlaySpec, DEFAULT_OVERLAY_SPEC

class OverlayGenerator:
    """Generate overlays with specified configuration."""
    
    def __init__(self, spec: Optional[OverlaySpec] = None):
        self.spec = spec or DEFAULT_OVERLAY_SPEC
        self.spec.validate()
    
    def create_overlay(
        self,
        image: np.ndarray,
        mask_true: np.ndarray,
        mask_pred: np.ndarray,
        save_path: Optional[Path] = None
    ) -> np.ndarray:
        """
        Create overlay visualization.
        
        Args:
            image: Original image (H, W, C)
            mask_true: Ground truth mask (H, W)
            mask_pred: Predicted mask (H, W)
            save_path: Optional path to save overlay
            
        Returns:
            Overlay image array
        """
        # Ensure inputs are numpy arrays
        image = np.asarray(image)
        mask_true = np.asarray(mask_true)
        mask_pred = np.asarray(mask_pred)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title("Original")
        axes[0].axis('off')
        
        # Prediction overlay
        axes[1].imshow(image)
        axes[1].imshow(
            mask_pred,
            alpha=self.spec.alpha,
            cmap=self.spec.colormap
        )
        axes[1].set_title("Prediction")
        axes[1].axis('off')
        
        # Error analysis
        if self.spec.error_colors:
            error_map = self._create_error_map(mask_true, mask_pred)
            axes[2].imshow(image)
            axes[2].imshow(error_map, alpha=0.5)
            axes[2].set_title("Error Analysis")
            axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            for fmt in self.spec.formats:
                output_path = save_path.with_suffix(f'.{fmt}')
                plt.savefig(
                    output_path,
                    dpi=self.spec.dpi,
                    bbox_inches='tight'
                )
        
        # Convert figure to array
        fig.canvas.draw()
        overlay_array = np.frombuffer(
            fig.canvas.tostring_rgb(),
            dtype=np.uint8
        )
        overlay_array = overlay_array.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )
        
        plt.close(fig)
        return overlay_array
    
    def _create_error_map(
        self,
        mask_true: np.ndarray,
        mask_pred: np.ndarray
    ) -> np.ndarray:
        """Create error visualization map."""
        h, w = mask_true.shape
        error_map = np.zeros((h, w, 4), dtype=np.uint8)
        
        # Calculate error types
        tp = (mask_true == 1) & (mask_pred == 1)
        fp = (mask_true == 0) & (mask_pred == 1)
        fn = (mask_true == 1) & (mask_pred == 0)
        
        # Apply colors
        error_map[tp] = self.spec.error_colors['true_positive']
        error_map[fp] = self.spec.error_colors['false_positive']
        error_map[fn] = self.spec.error_colors['false_negative']
        
        return error_map
```

## 10. Error Handling Strategy

### Comprehensive Error Handling

```python
# src/core/errors.py
"""Custom exceptions with helpful messages."""

class ConfigurationError(Exception):
    """Configuration related errors."""
    
    def __init__(self, message: str, suggestions: List[str] = None):
        self.message = message
        self.suggestions = suggestions or []
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format error message with suggestions."""
        msg = f"\n❌ Configuration Error: {self.message}"
        if self.suggestions:
            msg += "\n\n💡 Suggestions:"
            for suggestion in self.suggestions:
                msg += f"\n  • {suggestion}"
        return msg

class DependencyError(Exception):
    """Missing or incompatible dependency."""
    
    def __init__(self, package: str, required_version: str = None):
        self.package = package
        self.required_version = required_version
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format dependency error message."""
        msg = f"\n❌ Dependency Error: {self.package} not available"
        if self.required_version:
            msg += f" (requires {self.required_version})"
        msg += f"\n\n💡 To install: pip install {self.package}"
        msg += "\n💡 Or use fallback mode: --use-fallbacks"
        return msg

class MemoryError(Exception):
    """Memory related errors."""
    
    def __init__(self, required_mb: int, available_mb: int):
        self.required = required_mb
        self.available = available_mb
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format memory error message."""
        msg = f"\n❌ Memory Error: Insufficient memory"
        msg += f"\n  Required: {self.required}MB"
        msg += f"\n  Available: {self.available}MB"
        msg += "\n\n💡 Suggestions:"
        msg += "\n  • Reduce batch size"
        msg += "\n  • Enable gradient accumulation"
        msg += "\n  • Use smaller model"
        msg += "\n  • Enable CPU offloading"
        return msg
```

### Error Handling in Scripts

```python
# scripts/train.py (error handling section)
"""Training script with comprehensive error handling."""

import sys
import traceback
from pathlib import Path

from src.core.errors import ConfigurationError, DependencyError, MemoryError
from src.core.dependencies import DependencyManager

def main():
    """Main training entry point with error handling."""
    try:
        # Check dependencies upfront
        dm = DependencyManager()
        deps = dm.check_dependencies()
        
        if not all(deps.values()) and not args.use_fallbacks:
            missing = [k for k, v in deps.items() if not v]
            print(f"⚠️  Missing optional dependencies: {', '.join(missing)}")
            print("   Running with fallback implementations.")
            print("   For full performance, install with: pip install -r requirements/ml.txt")
        
        # Parse arguments
        args = parse_args()
        
        # Validate configuration
        try:
            config = load_and_validate_config(
                args.config,
                mode=args.validation_mode
            )
        except ConfigurationError as e:
            print(e)
            sys.exit(1)
        
        # Check memory before starting
        check_memory_requirements(config)
        
        # Run training
        run_training(config, args)
        
    except DependencyError as e:
        print(e)
        sys.exit(1)
        
    except MemoryError as e:
        print(e)
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        save_checkpoint_on_interrupt()
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("\n📋 Full traceback:")
        traceback.print_exc()
        
        print("\n💡 If this seems like a bug, please report it with:")
        print("   1. The full error message above")
        print("   2. Your configuration file")
        print("   3. Output of: python scripts/check_dependencies.py")
        sys.exit(1)
```

## 11. Implementation Guidance

### Common Pitfalls and Solutions

```markdown
# docs/IMPLEMENTATION_LESSONS.md
# Implementation Lessons Learned (Living Document)

## Lesson 1: Dependency Management
**Issue**: TorchMetrics API changes broke implementation
**Solution**: Always provide fallback implementations
**Implementation**:
```python
# ALWAYS structure imports like this:
try:
    from torchmetrics import Dice
    HAS_TORCHMETRICS = True
except ImportError:
    from .simple import SimpleDice as Dice
    HAS_TORCHMETRICS = False
```

## Lesson 2: Import Cycles
**Issue**: Circular imports between modules
**Solution**: Clear dependency hierarchy
**Implementation**:
- utils → core → data → models → training
- Never import from higher levels
- Use TYPE_CHECKING for type hints only

## Lesson 3: Platform Differences
**Issue**: CUDA/MPS/CPU behave differently
**Solution**: Platform-aware code with fallbacks
**Implementation**:
```python
def get_device(prefer_mps=True):
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif prefer_mps and torch.backends.mps.is_available():
        # Check for unsupported operations
        if not check_mps_operations():
            warnings.warn("MPS missing operations, falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device("cpu")
```

## Lesson 4: Configuration Complexity
**Issue**: Config validation too strict for development
**Solution**: Multiple validation modes
**Implementation**:
- MINIMAL: For testing (core fields only)
- PERMISSIVE: For development (warnings on issues)
- STRICT: For production (fail on any issue)

## Lesson 5: Memory Management
**Issue**: OOM errors during training
**Solution**: Proactive memory management
**Implementation**:
- Auto batch size adjustment
- Memory monitoring during training
- Gradient accumulation for large effective batches
```

### Platform-Specific Optimizations

```python
# src/utils/platform.py
"""Platform detection and optimization."""

import torch
import platform
import warnings
from typing import Dict, Any

class PlatformOptimizer:
    """Optimize settings for different platforms."""
    
    @staticmethod
    def get_platform_info() -> Dict[str, Any]:
        """Get current platform information."""
        info = {
            'system': platform.system(),
            'machine': platform.machine(),
            'python': platform.python_version(),
            'torch': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'mps_available': torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False,
            'cpu_count': torch.get_num_threads(),
        }
        
        if info['cuda_available']:
            info['cuda_version'] = torch.version.cuda
            info['gpu_name'] = torch.cuda.get_device_name(0)
            
        return info
    
    @staticmethod
    def optimize_settings(config: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize configuration for current platform."""
        info = PlatformOptimizer.get_platform_info()
        
        # Mac M1/M2 optimizations
        if info['machine'] == 'arm64' and info['system'] == 'Darwin':
            if info['mps_available']:
                config['compute']['accelerator'] = 'mps'
                # MPS doesn't support all operations
                config['compute']['fallback_to_cpu'] = True
            else:
                config['compute']['accelerator'] = 'cpu'
                # Use multiple threads on CPU
                torch.set_num_threads(min(8, info['cpu_count']))
        
        # CUDA optimizations
        elif info['cuda_available']:
            config['compute']['accelerator'] = 'cuda'
            # Enable mixed precision if GPU supports it
            if 'A100' in info.get('gpu_name', '') or 'V100' in info.get('gpu_name', ''):
                config['compute']['precision'] = 16
        
        # CPU optimizations
        else:
            config['compute']['accelerator'] = 'cpu'
            torch.set_num_threads(info['cpu_count'])
            # Reduce batch size for CPU
            config['training']['batch_size'] = min(
                config['training']['batch_size'],
                4
            )
        
        return config
```

## 12. K-Fold Cross-Validation Implementation

### Cross-Validation Orchestrator

```python
# src/core/cross_validation.py
"""K-fold cross-validation with error handling and memory management."""

from typing import List, Dict, Tuple, Callable, Optional
from pathlib import Path
import json
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold
import logging

from .errors import ConfigurationError
from .config import Config
from ..utils.memory import MemoryManager

logger = logging.getLogger(__name__)

class CrossValidator:
    """
    Orchestrates k-fold cross-validation with robust error handling.
    
    This is the primary enhancement after MVP completion.
    """
    
    def __init__(
        self,
        config: Config,
        run_dir: Path,
        use_stratified: bool = False,
        use_groups: bool = False
    ):
        self.config = config
        self.run_dir = run_dir
        self.use_stratified = use_stratified
        self.use_groups = use_groups
        self.fold_results = []
        
    def run(
        self,
        make_datamodule: Callable,
        make_model: Callable,
        make_trainer: Callable
    ) -> Tuple[List[Dict], Dict]:
        """
        Run k-fold cross-validation.
        
        Returns:
            (fold_results, aggregated_summary)
        """
        # Setup
        n_folds = self.config.cross_validation.folds
        seed = self.config.cross_validation.seed
        
        # Get data IDs and labels
        data_ids, labels, groups = self._load_data_info()
        
        # Create appropriate splitter
        splitter = self._create_splitter(n_folds, seed)
        
        # Check memory before starting
        mem_manager = MemoryManager()
        available_mb = mem_manager.get_available_memory()[0]
        if available_mb < 4000:  # Less than 4GB
            logger.warning(f"Low memory ({available_mb}MB). Adjusting settings...")
            self.config = self._adjust_for_low_memory(self.config)
        
        # Run each fold
        for fold_idx, (train_idx, val_idx) in enumerate(
            splitter.split(data_ids, labels, groups)
        ):
            try:
                fold_result = self._run_single_fold(
                    fold_idx=fold_idx,
                    train_idx=train_idx,
                    val_idx=val_idx,
                    data_ids=data_ids,
                    make_datamodule=make_datamodule,
                    make_model=make_model,
                    make_trainer=make_trainer
                )
                self.fold_results.append(fold_result)
                
            except Exception as e:
                logger.error(f"Fold {fold_idx} failed: {e}")
                if self.config.cross_validation.get('continue_on_error', False):
                    self.fold_results.append({
                        'fold': fold_idx,
                        'status': 'failed',
                        'error': str(e)
                    })
                else:
                    raise
        
        # Aggregate results
        summary = self._aggregate_results()
        
        # Save results
        self._save_results(summary)
        
        # Generate report
        self._generate_report(summary)
        
        return self.fold_results, summary
    
    def _run_single_fold(
        self,
        fold_idx: int,
        train_idx: np.ndarray,
        val_idx: np.ndarray,
        data_ids: np.ndarray,
        make_datamodule: Callable,
        make_model: Callable,
        make_trainer: Callable
    ) -> Dict:
        """Run single fold with comprehensive tracking."""
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Fold {fold_idx + 1}/{self.config.cross_validation.folds}")
        logger.info(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        logger.info(f"{'='*50}\n")
        
        # Create fold directory
        fold_dir = self.run_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        # Create components
        datamodule = make_datamodule(
            train_ids=data_ids[train_idx],
            val_ids=data_ids[val_idx],
            config=self.config
        )
        
        # Seed per fold for diversity
        model = make_model(
            config=self.config,
            seed=self.config.compute.seed + fold_idx
        )
        
        trainer = make_trainer(
            config=self.config,
            default_root_dir=fold_dir,
            fold=fold_idx
        )
        
        # Train
        trainer.fit(model, datamodule)
        
        # Validate
        val_results = trainer.validate(model, datamodule)[0]
        
        # Compile results
        fold_result = {
            'fold': fold_idx,
            'status': 'completed',
            'train_samples': len(train_idx),
            'val_samples': len(val_idx),
            'metrics': val_results,
            'best_epoch': trainer.current_epoch,
        }
        
        # Save fold results
        with open(fold_dir / 'results.json', 'w') as f:
            json.dump(fold_result, f, indent=2)
        
        return fold_result
    
    def _aggregate_results(self) -> Dict:
        """Aggregate metrics across folds."""
        
        # Filter successful folds
        successful_folds = [
            r for r in self.fold_results 
            if r.get('status') == 'completed'
        ]
        
        if not successful_folds:
            raise RuntimeError("No successful folds to aggregate")
        
        # Extract metrics
        metric_names = list(successful_folds[0]['metrics'].keys())
        
        summary = {
            'total_folds': self.config.cross_validation.folds,
            'successful_folds': len(successful_folds),
            'metrics': {}
        }
        
        for metric in metric_names:
            values = [r['metrics'][metric] for r in successful_folds]
            summary['metrics'][metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': values
            }
            
            # 95% confidence interval
            if len(values) > 1:
                sem = np.std(values) / np.sqrt(len(values))
                ci_95 = 1.96 * sem
                summary['metrics'][metric]['ci_95'] = float(ci_95)
        
        return summary
```

## 13. Dependency Diagnostic Tool

```python
# scripts/check_dependencies.py
"""Comprehensive dependency diagnostic tool."""

import sys
import importlib.util
from pathlib import Path
from typing import Dict, Tuple
import json

def check_dependency(package: str) -> Tuple[bool, str]:
    """Check if a package is available and get version."""
    spec = importlib.util.find_spec(package)
    if spec is None:
        return False, "Not installed"
    
    try:
        module = importlib.import_module(package)
        version = getattr(module, '__version__', 'Unknown version')
        return True, version
    except Exception as e:
        return False, f"Import error: {e}"

def main():
    """Run dependency diagnostics."""
    print("🔍 Checking Dependencies\n")
    print("=" * 60)
    
    # Core dependencies
    print("\n📦 Core Dependencies (Required):")
    core_deps = ['torch', 'numpy', 'pyyaml', 'pydantic', 'pillow']
    core_status = check_dependencies(core_deps)
    
    # ML dependencies
    print("\n🤖 ML Dependencies (Performance):")
    ml_deps = ['pytorch_lightning', 'segmentation_models_pytorch', 
               'torchmetrics', 'albumentations', 'timm']
    ml_status = check_dependencies(ml_deps)
    
    # Visualization dependencies
    print("\n📊 Visualization Dependencies:")
    viz_deps = ['matplotlib', 'seaborn', 'jinja2']
    viz_status = check_dependencies(viz_deps)
    
    # Platform information
    print("\n💻 Platform Information:")
    print_platform_info()
    
    # Recommendations
    print("\n💡 Recommendations:")
    provide_recommendations(core_status, ml_status, viz_status)
    
    # Save diagnostic report
    save_diagnostic_report(core_status, ml_status, viz_status)

def check_dependencies(deps: list) -> Dict[str, Tuple[bool, str]]:
    """Check a list of dependencies."""
    status = {}
    for dep in deps:
        available, version = check_dependency(dep)
        status[dep] = (available, version)
        
        if available:
            print(f"  ✅ {dep:<25} {version}")
        else:
            print(f"  ❌ {dep:<25} {version}")
    
    return status

def print_platform_info():
    """Print platform information."""
    import platform
    import torch
    
    print(f"  OS:        {platform.system()} {platform.release()}")
    print(f"  Python:    {platform.python_version()}")
    print(f"  PyTorch:   {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"  CUDA:      ✅ {torch.version.cuda}")
        print(f"  GPU:       {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CUDA:      ❌ Not available")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  MPS:       ✅ Available (Apple Silicon)")
    else:
        print(f"  MPS:       ❌ Not available")

def provide_recommendations(core_status, ml_status, viz_status):
    """Provide installation recommendations."""
    
    # Check core
    core_missing = [k for k, (v, _) in core_status.items() if not v]
    if core_missing:
        print(f"\n  ⚠️  Missing core dependencies: {', '.join(core_missing)}")
        print(f"     Install with: pip install -r requirements/base.txt")
    
    # Check ML
    ml_missing = [k for k, (v, _) in ml_status.items() if not v]
    if ml_missing:
        print(f"\n  ℹ️  Missing ML dependencies: {', '.join(ml_missing)}")
        print(f"     For full performance: pip install -r requirements/ml.txt")
        print(f"     The system will use fallback implementations.")
    
    # Check viz
    viz_missing = [k for k, (v, _) in viz_status.items() if not v]
    if viz_missing:
        print(f"\n  ℹ️  Missing visualization dependencies: {', '.join(viz_missing)}")
        print(f"     For enhanced visualizations: pip install -r requirements/viz.txt")

def save_diagnostic_report(core_status, ml_status, viz_status):
    """Save diagnostic report to file."""
    report = {
        'timestamp': str(Path.ctime(Path.cwd())),
        'core': {k: {'available': v[0], 'version': v[1]} 
                 for k, v in core_status.items()},
        'ml': {k: {'available': v[0], 'version': v[1]} 
               for k, v in ml_status.items()},
        'viz': {k: {'available': v[0], 'version': v[1]} 
                for k, v in viz_status.items()},
    }
    
    with open('dependency_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: dependency_report.json")

if __name__ == "__main__":
    main()
```

## 14. Architecture Decision Records

```markdown
# docs/ARCHITECTURE_DECISIONS.md

## ADR-001: Use SMP Instead of Custom Models
**Date**: 2025-01-15  
**Status**: Accepted

### Context
Need segmentation models quickly without reinventing the wheel.

### Decision
Use Segmentation Models PyTorch (SMP) with fallback simple UNet.

### Alternatives Considered
1. Custom implementations (too time-consuming)
2. MONAI (too medical-specific for MVP)
3. MMSegmentation (too complex)

### Consequences
- ✅ Fast implementation
- ✅ Proven models
- ✅ Good encoder variety via timm
- ❌ Extra dependency
- ❌ Less control over architecture

### Mitigation
Provide SimpleUNet fallback for when SMP unavailable.

---

## ADR-002: Three-Tier Testing Strategy
**Date**: 2025-01-15  
**Status**: Accepted

### Context
Need to test without installing heavy ML dependencies.

### Decision
Three testing tiers: unit (no deps), integration_minimal (core), integration_full (all).

### Consequences
- ✅ Can always run some tests
- ✅ Faster CI for PRs
- ✅ Better developer experience
- ❌ More test complexity

---

## ADR-003: Configuration Validation Modes
**Date**: 2025-01-15  
**Status**: Accepted

### Context
Strict validation hinders development iteration.

### Decision
Three validation modes: MINIMAL, PERMISSIVE, STRICT.

### Consequences
- ✅ Fast development iteration
- ✅ Production safety with STRICT
- ✅ Gradual configuration refinement
- ❌ Multiple code paths to maintain
```

## 15. Conclusion

This Software Design Document v3.0 incorporates comprehensive lessons learned from implementing the MVP through Week 2. Key improvements include:

1. **Robust Dependency Management** - Fallback implementations for all optional dependencies
2. **Three-Tier Testing Strategy** - Enables development without full stack
3. **Flexible Configuration Validation** - Multiple modes for different use cases
4. **Detailed Visualization Specifications** - No ambiguity in implementation
5. **Comprehensive Error Handling** - Helpful messages with solutions
6. **Memory Management** - Proactive monitoring and adjustment
7. **Platform Optimizations** - Automatic tuning for CUDA/MPS/CPU
8. **Living Documentation** - Implementation lessons continuously updated

The platform is now ready for:
- **Immediate**: K-fold cross-validation implementation
- **Next Sprint**: TTA and sliding window inference
- **Future**: Classification tasks and MONAI integration

This document serves as both a **blueprint for new features** and a **reference for solving common issues**, ensuring smooth development as the platform evolves.