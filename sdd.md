# Image Segmentation Platform — Software Design Document
*Version:* 2.0  
*Date:* 2025-01-15

## Executive Summary
Build a **focused, production-ready** segmentation platform using **proven libraries** (SMP, PyTorch Lightning) while maintaining clean interfaces for future extensibility. Start with segmentation only, clear path to classification and multimodal later.

**Key Changes from v1.0:**
- Focus on segmentation only (classification deferred)
- Leverage SMP + PyTorch Lightning instead of hand-rolling
- Start with single train/val split (k-fold as immediate next step)
- Strong emphasis on documentation and code clarity

## 1. Purpose & Scope

### Current Scope (MVP)
- **Task:** Semantic segmentation only
- **Models:** U-Net/U-Net++/DeepLabV3+ via Segmentation Models PyTorch (SMP)
- **Data:** Local PNG/JPEG images with mask labels
- **Validation:** Single train/val split with fixed seed
- **Target:** DRIVE dataset (retinal vessel segmentation)
- **Platform:** M1 MacBook (CPU), with CUDA support ready

### Immediate Next Phase (v2.1)
- **5-fold cross-validation** (implementation approach documented below)
- **Additional backbones** via timm integration
- **TTA inference** (horizontal flip initially)

### Future Scope (v3+)
- Classification tasks (see Section 22)
- MONAI integration for medical imaging
- Multimodal (image + tabular) 

## 2. Core Technical Stack

### Foundation Libraries
- **PyTorch**: Core deep learning framework
- **PyTorch Lightning**: Training orchestration, eliminates boilerplate
- **Segmentation Models PyTorch (SMP)**: Pre-built architectures
- **TorchMetrics**: Standardized metric computation
- **Albumentations**: 2D augmentations
- **timm**: Encoder backbones

### Infrastructure
- **Python 3.11+**
- **venv** for environment management  
- **Git** for version control
- **SQLite** for lightweight metadata tracking
- **YAML** for configuration
- **Docker** ready (future)

### Future Additions
- **MONAI**: Medical I/O, 3D transforms, sliding window inference
- **TorchIO**: 3D medical augmentations
- **MLflow/W&B**: Experiment tracking (optional layer)

## 3. Architecture: Five Stable Interfaces

The platform is built around **5 minimal, stable contracts** that allow component swapping without rippling changes:

### 3.1 DataModule Contract
```python
class DataModule(Protocol):
    """Owns data loading, splits, transforms"""
    def setup(self, stage: str) -> None: ...
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> DataLoader: ...
    def test_dataloader(self) -> Optional[DataLoader]: ...
```

### 3.2 Model Contract  
```python
class Model(nn.Module):
    """Pure PyTorch model"""
    def forward(self, x: Tensor) -> Tensor: ...
    def predict_step(self, x: Tensor) -> Tensor:
        """Optional: for sliding window or TTA"""
        return self.forward(x)
```

### 3.3 Loss Contract
```python
class Loss(Protocol):
    """Any callable loss function"""
    def __call__(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor: ...
```

### 3.4 Metrics Contract
```python
class Metrics(Protocol):
    """TorchMetrics-compatible"""
    def update(self, pred: Tensor, target: Tensor) -> None: ...
    def compute(self) -> Dict[str, Tensor]: ...
    def reset(self) -> None: ...
```

### 3.5 Trainer Contract
```python
class Trainer(Protocol):
    """Lightning trainer or compatible"""
    def fit(self, model, datamodule, callbacks=None) -> None: ...
    def validate(self, model, datamodule) -> List[Dict]: ...
    def test(self, model, datamodule) -> List[Dict]: ...
```

**These interfaces remain stable** when switching between tasks, models, or datasets.

## 4. Configuration Schema

### MVP Configuration (Single Split)
```yaml
# config.yaml - Exhaustively commented
project_name: "drive_vessel_segmentation"
task: "segmentation"  # Future: "classification" | "multimodal"

# Dataset configuration
dataset:
  name: "DRIVE"
  images_dir: "./data/drive/images"
  masks_dir: "./data/drive/masks"  
  # For classification: labels_file: "./labels.csv"
  
  # Image file pattern
  image_suffix: ".png"
  mask_suffix: "_mask.png"
  
  # Train/val split  
  split:
    type: "random"  # Future: "stratified", "group"
    val_ratio: 0.2
    seed: 42

# Model configuration  
model:
  architecture: "Unet"  # Options: "Unet", "UnetPlusPlus", "DeepLabV3Plus"
  encoder: "resnet34"   # Any timm encoder
  encoder_weights: "imagenet"
  in_channels: 3
  classes: 1  # Binary segmentation

# Training configuration
training:
  epochs: 50
  batch_size: 8
  learning_rate: 1e-4
  optimizer: "adamw"
  scheduler:
    type: "cosine"
    min_lr: 1e-6
  
  # Loss function
  loss:
    type: "dice"  # Options: "dice", "tversky", "bce", "focal"
    params:
      smooth: 1.0
  
  # Metrics to track
  metrics:
    - "dice"
    - "iou" 
    - "precision"
    - "recall"

# Augmentations (Albumentations format)
augmentations:
  train:
    - name: "RandomRotate90"
      params: {p: 0.5}
    - name: "HorizontalFlip"
      params: {p: 0.5}
    - name: "RandomBrightnessContrast"
      params: {brightness_limit: 0.2, contrast_limit: 0.2, p: 0.5}
  val: []  # No augmentation for validation

# Inference settings
inference:
  mode: "standard"  # Future: "tta", "sliding_window"
  # tta:
  #   transforms: ["hflip"]  # Future: ["hflip", "vflip", "rot90"]
  # sliding_window:
  #   window_size: [256, 256]
  #   overlap: 0.5

# Output configuration
output:
  dir: "./runs"
  save_overlays: true
  save_predictions: true
  checkpoint:
    monitor: "val_loss"
    mode: "min"
    save_best_only: true

# Compute settings
compute:
  accelerator: "auto"  # "cpu", "gpu", "mps", "auto"
  devices: 1
  precision: 32  # 16 for mixed precision
  deterministic: true
  seed: 42

# Logging
logging:
  level: "INFO"
  save_tensorboard: false  # Future: true
```

### Future: K-Fold Configuration Addition
```yaml
# Additional section for k-fold CV (v2.1)
cross_validation:
  enabled: true
  folds: 5
  stratify: true  # For classification
  group_by: "patient_id"  # For medical data
  seed: 42
```

## 5. Repository Structure

```
segmentation-platform/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Config validation with Pydantic
│   │   ├── trainer.py         # Lightning trainer wrapper
│   │   └── registry.py        # Component registry
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── datamodule.py      # Lightning DataModule
│   │   ├── dataset.py         # PyTorch Dataset
│   │   ├── transforms.py      # Augmentation builders
│   │   └── readers/           # Future: DICOM, NIfTI readers
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── segmentation.py    # SMP model wrapper
│   │   ├── lightning_module.py # LightningModule implementation
│   │   └── architectures/     # Future: custom architectures
│   │
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── dice.py            # Dice loss variants
│   │   ├── tversky.py         # Tversky loss
│   │   └── compound.py        # Combined losses
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   └── segmentation.py    # TorchMetrics wrappers
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict.py         # Single image/batch prediction
│   │   ├── tta.py             # Test-time augmentation
│   │   └── sliding_window.py  # Future: sliding window
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── overlays.py        # Mask overlays
│   │   └── metrics_plot.py    # Training curves
│   │
│   └── utils/
│       ├── __init__.py
│       ├── io.py              # File I/O helpers
│       ├── logging.py         # Logging setup
│       └── reproducibility.py # Seed management
│
├── configs/
│   ├── base_config.yaml       # Default configuration
│   ├── drive.yaml             # DRIVE dataset specific
│   └── examples/              # Example configs
│
├── scripts/
│   ├── train.py               # Main training script
│   ├── predict.py             # Inference script
│   ├── evaluate.py            # Evaluation script
│   └── visualize_results.py   # Results visualization
│
├── tests/
│   ├── unit/                  # Unit tests per module
│   ├── integration/           # End-to-end tests
│   └── fixtures/              # Test data
│
├── runs/                      # Experiment outputs (gitignored)
│
├── docs/
│   ├── README.md              # Comprehensive documentation
│   ├── QUICKSTART.md          # 5-minute getting started
│   ├── CONFIG_GUIDE.md        # Configuration deep dive
│   ├── ARCHITECTURE.md        # Technical architecture
│   └── EXTENDING.md           # How to add components
│
├── requirements.txt           # Pinned dependencies
├── setup.py                   # Package setup
└── .gitignore
```

## 6. Documentation Requirements

### 6.1 Code Documentation Standards
**Every module, class, and public method must include:**

```python
class SegmentationDataModule(LightningDataModule):
    """
    PyTorch Lightning DataModule for segmentation tasks.
    
    This class handles all data-related operations including loading,
    splitting, transformation, and batching. It ensures reproducible
    data splits and consistent preprocessing across training phases.
    
    Args:
        config (Dict): Configuration dictionary containing:
            - dataset.images_dir: Path to images directory
            - dataset.masks_dir: Path to masks directory
            - dataset.split: Split configuration
            - augmentations: Augmentation configurations
            - training.batch_size: Batch size for dataloaders
        
    Example:
        >>> config = load_config("configs/drive.yaml")
        >>> dm = SegmentationDataModule(config)
        >>> dm.setup("fit")
        >>> train_loader = dm.train_dataloader()
    
    Note:
        The DataModule maintains separate transforms for train/val/test
        to ensure proper augmentation strategies per phase.
    """
```

### 6.2 README Structure
The README must include:

1. **Quick Start** (5 minutes to first run)
   - Installation
   - Download example data
   - Run training
   - View results

2. **Detailed Usage**
   - Configuration system explanation
   - All CLI commands and options
   - Input data format requirements
   - Output structure explanation

3. **Configuration Deep Dive**
   - Every configuration option explained
   - Common configuration patterns
   - Performance tuning guide

4. **Architecture Overview**
   - Interface contracts
   - Data flow diagram
   - Extension points

5. **Extending the Platform**
   - Adding new models
   - Adding new losses
   - Adding new metrics
   - Adding new data formats

6. **Troubleshooting**
   - Common issues and solutions
   - Performance optimization tips
   - Debug mode usage

### 6.3 Inline Comment Requirements
```python
def create_augmentation_pipeline(config: Dict, phase: str) -> A.Compose:
    """Create Albumentations pipeline from config."""
    
    # Early return for validation/test phases - no augmentation needed
    if phase != "train" or "augmentations" not in config:
        return A.Compose([A.Normalize()])
    
    transforms = []
    
    # Build transform list from config
    # Each transform is specified as {name: str, params: dict}
    for aug_config in config["augmentations"].get(phase, []):
        # Dynamically get transform class from Albumentations
        transform_class = getattr(A, aug_config["name"])
        
        # Instantiate with provided parameters
        # Default probability to 1.0 if not specified
        params = aug_config.get("params", {})
        if "p" not in params:
            params["p"] = 1.0
            
        transforms.append(transform_class(**params))
    
    # Always end with normalization
    transforms.append(A.Normalize())
    
    return A.Compose(transforms)
```

## 7. Output Structure

### Single Run Output (MVP)
```
runs/
└── drive_2025-01-15_093045_a3f2b1/
    ├── config.yaml              # Exact config used
    ├── environment.json         # Python/package versions
    ├── 
    ├── checkpoints/
    │   └── best.ckpt           # Best model weights
    │
    ├── predictions/            # Optional: raw predictions
    │   ├── val_image_001.npy
    │   └── ...
    │
    ├── overlays/               # Visual overlays
    │   ├── val_image_001.png
    │   └── ...
    │
    ├── metrics/
    │   ├── train_metrics.csv   # Epoch-by-epoch metrics
    │   ├── val_metrics.csv
    │   └── final_metrics.json  # Final summary
    │
    ├── logs/
    │   ├── train.log           # Detailed training log
    │   └── tensorboard/        # Future: TB logs
    │
    └── report.md               # Human-readable summary
```

### K-Fold Output Structure (v2.1)
```
runs/
└── drive_kfold_2025-01-20_143022_b4c3d2/
    ├── config.yaml
    ├── environment.json
    ├── 
    ├── fold_0/
    │   ├── checkpoints/
    │   ├── metrics/
    │   └── overlays/
    ├── fold_1/
    │   └── ...
    ├── fold_2/
    │   └── ...
    ├── fold_3/
    │   └── ...
    ├── fold_4/
    │   └── ...
    │
    ├── aggregated/
    │   ├── metrics_summary.json  # Mean ± std across folds
    │   ├── metrics_per_fold.csv  # Detailed per-fold
    │   └── statistical_tests.json # Optional: significance tests
    │
    └── report.md                 # Includes CV analysis
```

## 8. CLI Interface

### MVP Commands
```bash
# Training
python scripts/train.py \
    --config configs/drive.yaml \
    --name "baseline_experiment" \
    --gpu 0

# Prediction
python scripts/predict.py \
    --checkpoint runs/drive_2025/checkpoints/best.ckpt \
    --input data/test_images/ \
    --output predictions/

# Evaluation
python scripts/evaluate.py \
    --checkpoint runs/drive_2025/checkpoints/best.ckpt \
    --test-dir data/test/ \
    --output eval_results.json

# Visualization
python scripts/visualize_results.py \
    --run-dir runs/drive_2025/ \
    --output visualizations/
```

### All Scripts Must:
- Accept `--help` with detailed descriptions
- Support `--config` for full configuration
- Allow command-line overrides of config values
- Log all operations with configurable verbosity
- Return appropriate exit codes

## 9. Testing Strategy

### Unit Tests (Minimal for MVP)
```python
def test_dice_loss():
    """Test Dice loss computation."""
    pred = torch.sigmoid(torch.randn(2, 1, 32, 32))
    target = torch.randint(0, 2, (2, 1, 32, 32)).float()
    loss = DiceLoss()(pred, target)
    assert 0 <= loss <= 1
```

### Integration Tests (Priority)
```python
def test_training_single_epoch():
    """Test that training runs for one epoch."""
    config = load_test_config()
    config["training"]["epochs"] = 1
    
    # Use Lightning's fast_dev_run for quick validation
    trainer = Trainer(fast_dev_run=True)
    model = create_model(config)
    datamodule = create_datamodule(config)
    
    trainer.fit(model, datamodule)
    
    # Check outputs exist
    assert Path("runs").exists()
    assert len(list(Path("runs").glob("*/checkpoints/*.ckpt"))) > 0
```

## 10. Implementation Priorities

### Week 1: Core Pipeline
1. **DataModule** with DRIVE dataset
2. **SMP Model wrapper** with Lightning
3. **Basic training script**
4. **Single train/val split**
5. **Dice loss and metric**

### Week 2: Full MVP
1. **All losses** (Dice, Tversky, BCE, Focal)
2. **All metrics** (IoU, Precision, Recall)
3. **Overlay visualization**
4. **Comprehensive config system**
5. **README and documentation**

### Week 3: Testing and Polish
1. **Integration tests**
2. **Performance profiling**
3. **Code cleanup and comments**
4. **Example notebooks** (optional)

## 11. K-Fold Cross-Validation (Next Immediate Enhancement)

### Implementation Plan
```python
# src/core/cross_validation.py
"""
K-fold cross-validation orchestrator.

This module provides the primary enhancement after MVP completion.
It wraps the single-split training pipeline to run multiple folds
and aggregate results.
"""

from sklearn.model_selection import GroupKFold, StratifiedKFold
import pytorch_lightning as pl
from pathlib import Path
import json
import numpy as np

def cross_validate(
    config: Dict,
    make_datamodule: Callable,
    make_model: Callable,
    make_trainer: Callable,
    run_root: Path
) -> Tuple[List[Dict], Dict]:
    """
    Run k-fold cross-validation.
    
    Args:
        config: Configuration dictionary with CV settings
        make_datamodule: Factory function for DataModule
        make_model: Factory function for model
        make_trainer: Factory function for trainer
        run_root: Root directory for outputs
    
    Returns:
        Tuple of (per_fold_results, aggregated_summary)
    
    Example:
        >>> results, summary = cross_validate(
        ...     config, 
        ...     make_datamodule,
        ...     make_model,
        ...     make_trainer,
        ...     Path("runs/cv_experiment")
        ... )
        >>> print(f"Mean Dice: {summary['dice_mean']:.3f}")
    """
    
    cv_config = config.get("cross_validation", {})
    n_folds = cv_config.get("folds", 5)
    seed = cv_config.get("seed", 42)
    
    # Setup fold splitter based on configuration
    if cv_config.get("group_by"):
        # Group-based splitting (e.g., by patient)
        splitter = GroupKFold(n_splits=n_folds)
        groups = load_groups(config)  # Load group assignments
    else:
        # Standard k-fold
        splitter = StratifiedKFold(
            n_splits=n_folds, 
            shuffle=True, 
            random_state=seed
        )
        groups = None
    
    # Collect dataset IDs
    dataset_ids = load_dataset_ids(config)
    labels = load_labels(config) if cv_config.get("stratify") else None
    
    # Run each fold
    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(dataset_ids, labels, groups)
    ):
        print(f"\n{'='*50}")
        print(f"Training Fold {fold_idx + 1}/{n_folds}")
        print(f"{'='*50}\n")
        
        # Create fold-specific components
        fold_dir = run_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        
        datamodule = make_datamodule(
            train_ids=dataset_ids[train_idx],
            val_ids=dataset_ids[val_idx],
            config=config
        )
        
        model = make_model(config=config, seed=seed + fold_idx)
        
        trainer = make_trainer(
            config=config,
            default_root_dir=fold_dir,
            fold=fold_idx
        )
        
        # Train this fold
        trainer.fit(model, datamodule)
        
        # Evaluate on validation set
        val_metrics = trainer.validate(model, datamodule)[0]
        
        # Store results
        fold_result = {
            "fold": fold_idx,
            "train_samples": len(train_idx),
            "val_samples": len(val_idx),
            **val_metrics
        }
        fold_results.append(fold_result)
        
        # Save fold-specific metrics
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(fold_result, f, indent=2)
    
    # Aggregate results across folds
    summary = aggregate_cv_metrics(fold_results)
    
    # Save aggregated results
    with open(run_root / "cv_summary.json", "w") as f:
        json.dump({
            "per_fold": fold_results,
            "aggregated": summary,
            "config": cv_config
        }, f, indent=2)
    
    # Generate CV report
    generate_cv_report(fold_results, summary, run_root)
    
    return fold_results, summary


def aggregate_cv_metrics(fold_results: List[Dict]) -> Dict:
    """
    Aggregate metrics across CV folds.
    
    Computes mean, std, min, max, and 95% CI for each metric.
    """
    # Extract metric names (exclude metadata fields)
    exclude_fields = {"fold", "train_samples", "val_samples"}
    metric_names = [
        k for k in fold_results[0].keys() 
        if k not in exclude_fields
    ]
    
    summary = {}
    for metric in metric_names:
        values = np.array([r[metric] for r in fold_results])
        
        summary[f"{metric}_mean"] = float(np.mean(values))
        summary[f"{metric}_std"] = float(np.std(values))
        summary[f"{metric}_min"] = float(np.min(values))
        summary[f"{metric}_max"] = float(np.max(values))
        
        # 95% confidence interval
        sem = np.std(values) / np.sqrt(len(values))
        ci_95 = 1.96 * sem
        summary[f"{metric}_ci95"] = float(ci_95)
    
    return summary
```

### Configuration for K-Fold
```yaml
# When ready to enable k-fold (v2.1)
cross_validation:
  enabled: true
  folds: 5
  seed: 42
  
  # Optional: stratification for imbalanced data
  stratify: true
  stratify_by: "lesion_type"  # Column name in labels
  
  # Optional: group-based splitting to prevent data leakage
  group_by: "patient_id"  # Ensures same patient not in train & val
  
  # Optional: nested CV for hyperparameter tuning
  nested:
    enabled: false
    inner_folds: 3
```

## 12. Performance & Profiling

### Built-in Profiling
```python
# Lightning provides profiling out of the box
trainer = Trainer(
    profiler="simple",  # or "advanced" for detailed profiling
    max_epochs=10
)

# Results automatically logged showing:
# - Time per epoch
# - Time per batch
# - Data loading time
# - Model forward/backward time
```

### Memory Optimization
- Gradient accumulation for large effective batch sizes
- Mixed precision training (fp16) when supported
- Automatic batch size finder (Lightning feature)

## 13. Migration Path to Classification

### What Stays the Same
- **Infrastructure**: Lightning trainer, callbacks, logging, checkpointing
- **Data Pipeline**: DataModule pattern, transforms, caching
- **Backbones**: Same encoders from timm/torchvision
- **Config System**: Same YAML structure with task switch
- **Output Structure**: Same directory layout
- **Testing Framework**: Same test patterns

### What Changes
| Component | Segmentation | Classification |
|-----------|--------------|----------------|
| **Model Head** | U-Net decoder | Global pool + Linear |
| **Loss** | Dice/Tversky | CrossEntropy/Focal |
| **Metrics** | IoU/Dice | Accuracy/AUROC/F1 |
| **Labels** | Mask images | CSV with classes |
| **Inference** | May need sliding window | Single forward pass |
| **Augmentations** | Geometric on image+mask | Geometric on image only |

### Example Classification Config (Future)
```yaml
task: "classification"

dataset:
  labels_file: "./labels.csv"  # Instead of masks_dir
  label_column: "diagnosis"
  
model:
  architecture: "resnet50"  # Or "efficientnet_b0", "vit_base"
  num_classes: 4
  pretrained: true
  
training:
  loss:
    type: "cross_entropy"  # Or "focal" for imbalanced
    params:
      weight: [1.0, 2.0, 2.0, 3.0]  # Class weights
      
  metrics:
    - "accuracy"
    - "auroc"
    - "f1_macro"
    - "confusion_matrix"
```

## 14. Future Enhancements

### Near-term (v2.1 - v2.3)
1. **K-fold cross-validation** (detailed above)
2. **Test-time augmentation** (horizontal flip → rotation → multi-scale)
3. **Sliding window inference** for large images
4. **Additional losses**: Focal, Boundary, Hausdorff
5. **Experiment tracking**: MLflow/W&B integration

### Medium-term (v3.0)
1. **MONAI Integration**:
   - Medical image I/O (NIfTI, DICOM)
   - 3D transforms and models
   - Medical-specific metrics (HD95, Surface Dice)
   - Sliding window inference utilities

2. **Classification Support**:
   - Shared codebase with segmentation
   - Multi-label and multi-class
   - Class activation maps

### Long-term (v4.0+)
1. **Multimodal**:
   - Image + tabular fusion
   - Early/late fusion strategies
   - Cross-attention mechanisms

2. **Advanced Features**:
   - AutoML for hyperparameter search
   - Neural architecture search
   - Self-supervised pretraining
   - Active learning loops

## 15. Quality Assurance

### Code Quality Standards
- Type hints for all public functions
- Docstrings following Google style
- Maximum line length: 100 characters
- McCabe complexity < 10
- Test coverage > 80% for core modules

### Review Checklist
- [ ] Config validates without errors
- [ ] Training runs to completion
- [ ] Metrics are logged correctly
- [ ] Checkpoints are saved
- [ ] Visualizations generated
- [ ] Documentation updated
- [ ] Tests pass

## 16. Success Metrics

### MVP Success Criteria
1. **Functional**: DRIVE segmentation pipeline runs end-to-end
2. **Performance**: Dice score > 0.75 on DRIVE test set
3. **Speed**: Training completes in < 1 hour on M1 MacBook
4. **Documentation**: README enables new user success in < 30 minutes
5. **Extensibility**: New model addition requires < 50 lines of code

### v2.1 Success Criteria
1. **K-fold CV** produces statistically meaningful results
2. **Documentation** includes 3+ complete examples
3. **Test coverage** > 80%
4. **Community**: First external contributor

## 17. Risk Mitigation

| Risk | Mitigation |
|------|------------|
| **Overfitting to small datasets** | Strong augmentation, cross-validation |
| **Library version conflicts** | Pin all dependencies, use lock files |
| **Memory issues on large images** | Sliding window, gradient accumulation |
| **Reproducibility concerns** | Fixed seeds, deterministic mode, config tracking |
| **Performance bottlenecks** | Profiling from day 1, mixed precision ready |

## 18. Development Guidelines

### Git Workflow
```bash
# Feature branch workflow
git checkout -b feature/add-tversky-loss
# Make changes
git add -p  # Review changes carefully
git commit -m "feat: add Tversky loss with alpha/beta parameters"
git push origin feature/add-tversky-loss
# Create PR with tests and documentation
```

### Commit Message Format
```
<type>: <description>

[optional body]

Type: feat|fix|docs|style|refactor|test|chore
```

### PR Requirements
- Passes all tests
- Includes documentation updates
- Has meaningful commit messages
- Reviewed by at least one person

## 19. Conclusion

This design provides a **production-ready segmentation platform** that:
1. **Leverages proven tools** (SMP, Lightning) instead of reinventing
2. **Maintains clean interfaces** for future extensibility  
3. **Focuses on immediate value** (segmentation) with clear upgrade paths
4. **Emphasizes documentation** for team scalability
5. **Builds on solid foundations** ready for MONAI, classification, and multimodal

The platform can be implemented in **2-3 weeks** for MVP, with k-fold CV and enhancements following immediately after validation.

## 20. Appendix A: Code Examples

### A.1 Minimal Training Script
```python
# scripts/train.py
"""
Main training script for segmentation platform.

This script orchestrates the entire training pipeline from config
loading through model training to results visualization.
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint, 
    EarlyStopping,
    LearningRateMonitor
)
from pytorch_lightning.loggers import CSVLogger

from src.core.config import load_and_validate_config
from src.data.datamodule import SegmentationDataModule
from src.models.lightning_module import SegmentationModel
from src.utils.reproducibility import set_global_seed
from src.utils.logging import setup_logging


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
    args = parser.parse_args()
    
    # Load and validate configuration
    config = load_and_validate_config(args.config)
    
    # Setup experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = args.name or f"{config['project_name']}_{timestamp}"
    run_dir = Path(config['output']['dir']) / exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(run_dir / "train.log", level=config['logging']['level'])
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training: {exp_name}")
    
    # Save configuration
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Set random seeds for reproducibility
    seed = config['compute']['seed']
    set_global_seed(seed)
    logger.info(f"Random seed set to: {seed}")
    
    # Save environment info
    save_environment_info(run_dir / "environment.json")
    
    # Create data module
    logger.info("Setting up data module...")
    datamodule = SegmentationDataModule(config)
    datamodule.setup("fit")
    
    # Log data statistics
    logger.info(f"Training samples: {len(datamodule.train_dataset)}")
    logger.info(f"Validation samples: {len(datamodule.val_dataset)}")
    
    # Create model
    logger.info("Creating model...")
    model = SegmentationModel(config)
    logger.info(f"Model: {config['model']['architecture']} "
                f"with {config['model']['encoder']} encoder")
    
    # Setup callbacks
    callbacks = [
        # Save best model based on validation loss
        ModelCheckpoint(
            dirpath=run_dir / "checkpoints",
            filename="best",
            monitor=config['output']['checkpoint']['monitor'],
            mode=config['output']['checkpoint']['mode'],
            save_best_only=config['output']['checkpoint']['save_best_only'],
            verbose=True
        ),
        
        # Learning rate monitoring
        LearningRateMonitor(logging_interval='epoch'),
    ]
    
    # Add early stopping if configured
    if config['training'].get('early_stopping'):
        callbacks.append(
            EarlyStopping(
                monitor=config['training']['early_stopping']['monitor'],
                patience=config['training']['early_stopping']['patience'],
                mode=config['training']['early_stopping']['mode'],
                verbose=True
            )
        )
    
    # Setup trainer
    logger.info("Initializing trainer...")
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        max_epochs=config['training']['epochs'],
        accelerator=config['compute']['accelerator'],
        devices=config['compute']['devices'],
        precision=config['compute']['precision'],
        deterministic=config['compute']['deterministic'],
        callbacks=callbacks,
        logger=CSVLogger(run_dir / "metrics"),
        enable_progress_bar=True,
        log_every_n_steps=10,
        
        # Optional: gradient accumulation for effective larger batches
        accumulate_grad_batches=config['training'].get(
            'accumulate_grad_batches', 1
        ),
        
        # Optional: mixed precision training
        amp_backend='native' if config['compute']['precision'] == 16 else None,
    )
    
    # Train model
    logger.info("Starting training...")
    trainer.fit(
        model=model,
        datamodule=datamodule,
        ckpt_path=args.resume
    )
    
    # Run validation on best model
    logger.info("Running final validation...")
    trainer.validate(
        model=model,
        datamodule=datamodule,
        ckpt_path="best"
    )
    
    # Generate visualizations if requested
    if config['output'].get('save_overlays'):
        logger.info("Generating overlays...")
        from src.visualization.overlays import generate_overlays
        generate_overlays(
            model=model,
            datamodule=datamodule,
            output_dir=run_dir / "overlays",
            device=trainer.device
        )
    
    # Generate final report
    logger.info("Generating report...")
    from src.reporting.report import generate_training_report
    generate_training_report(
        run_dir=run_dir,
        config=config
    )
    
    logger.info(f"Training complete! Results saved to: {run_dir}")


def save_environment_info(output_path: Path):
    """Save environment information for reproducibility."""
    import torch
    import torchvision
    import segmentation_models_pytorch as smp
    import pytorch_lightning as pl
    import albumentations as A
    
    env_info = {
        "python": sys.version,
        "pytorch": torch.__version__,
        "torchvision": torchvision.__version__,
        "lightning": pl.__version__,
        "smp": smp.__version__,
        "albumentations": A.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    
    with open(output_path, "w") as f:
        json.dump(env_info, f, indent=2)


if __name__ == "__main__":
    main()
```

### A.2 Lightning Module Implementation
```python
# src/models/lightning_module.py
"""
PyTorch Lightning module for segmentation.

This module encapsulates the model, loss, metrics, and training logic
in a reusable Lightning module that handles all aspects of training.
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torchmetrics import Dice, JaccardIndex, Precision, Recall

from src.losses import get_loss
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SegmentationModel(pl.LightningModule):
    """
    Lightning module for segmentation tasks.
    
    This class wraps an SMP model with Lightning training logic,
    handling loss computation, metric tracking, and optimization.
    
    Args:
        config: Configuration dictionary containing model, training,
                and loss specifications
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Create model from SMP
        model_config = config['model']
        self.model = getattr(smp, model_config['architecture'])(
            encoder_name=model_config['encoder'],
            encoder_weights=model_config.get('encoder_weights', 'imagenet'),
            in_channels=model_config.get('in_channels', 3),
            classes=model_config.get('classes', 1),
        )
        
        # Setup loss function
        loss_config = config['training']['loss']
        self.loss_fn = get_loss(loss_config['type'], loss_config.get('params', {}))
        
        # Setup metrics
        metrics_config = config['training'].get('metrics', ['dice'])
        self.setup_metrics(metrics_config)
        
        # Training configuration
        self.learning_rate = config['training']['learning_rate']
        self.optimizer_name = config['training'].get('optimizer', 'adamw')
        self.scheduler_config = config['training'].get('scheduler', {})
        
    def setup_metrics(self, metric_names):
        """Initialize TorchMetrics metrics."""
        self.train_metrics = {}
        self.val_metrics = {}
        
        for name in metric_names:
            if name == 'dice':
                metric = Dice(num_classes=2, average='micro')
            elif name == 'iou':
                metric = JaccardIndex(num_classes=2, average='micro')
            elif name == 'precision':
                metric = Precision(num_classes=2, average='micro')
            elif name == 'recall':
                metric = Recall(num_classes=2, average='micro')
            else:
                logger.warning(f"Unknown metric: {name}, skipping")
                continue
            
            # Clone for train and val to maintain separate states
            self.train_metrics[name] = metric.clone()
            self.val_metrics[name] = metric.clone()
    
    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        images, masks = batch
        
        # Forward pass
        logits = self(images)
        
        # Calculate loss
        loss = self.loss_fn(logits, masks)
        
        # Calculate and log metrics
        preds = torch.sigmoid(logits)
        preds_binary = (preds > 0.5).float()
        
        for name, metric in self.train_metrics.items():
            metric(preds_binary, masks)
            self.log(f'train/{name}', metric, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        images, masks = batch
        
        # Forward pass
        logits = self(images)
        
        # Calculate loss
        loss = self.loss_fn(logits, masks)
        
        # Calculate and log metrics
        preds = torch.sigmoid(logits)
        preds_binary = (preds > 0.5).float()
        
        for name, metric in self.val_metrics.items():
            metric(preds_binary, masks)
            self.log(f'val/{name}', metric, on_step=False, on_epoch=True, prog_bar=True)
        
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        
        # Setup optimizer
        if self.optimizer_name == 'adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.learning_rate
            )
        elif self.optimizer_name == 'adamw':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.learning_rate,
                weight_decay=self.config['training'].get('weight_decay', 1e-4)
            )
        elif self.optimizer_name == 'sgd':
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.config['training'].get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        
        # Return optimizer only if no scheduler
        if not self.scheduler_config:
            return optimizer
        
        # Setup scheduler
        scheduler_type = self.scheduler_config['type']
        
        if scheduler_type == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,
                eta_min=self.scheduler_config.get('min_lr', 1e-6)
            )
        elif scheduler_type == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config.get('step_size', 10),
                gamma=self.scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.scheduler_config.get('factor', 0.5),
                patience=self.scheduler_config.get('patience', 5)
            )
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': 'val/loss',
                }
            }
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type}")
        
        return [optimizer], [scheduler]
    
    def predict_step(self, batch, batch_idx):
        """
        Prediction step for inference.
        
        Can be extended for TTA or sliding window inference.
        """
        images = batch if isinstance(batch, torch.Tensor) else batch[0]
        
        # Standard prediction
        logits = self(images)
        preds = torch.sigmoid(logits)
        
        return (preds > 0.5).float()
```

## 21. Appendix B: Sample README Structure

```markdown
# Image Segmentation Platform

A production-ready deep learning platform for image segmentation using PyTorch Lightning and Segmentation Models PyTorch.

## 🚀 Quick Start (5 minutes)

### Installation
```bash
# Clone repository
git clone https://github.com/yourorg/segmentation-platform
cd segmentation-platform

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Example Data
```bash
# Download DRIVE dataset
python scripts/download_drive.py --output data/drive
```

### Train Your First Model
```bash
# Train U-Net on DRIVE dataset
python scripts/train.py --config configs/drive.yaml --name my_first_run

# Monitor training (metrics are auto-logged)
# Results will be in runs/my_first_run/
```

### View Results
```bash
# Generate overlay visualizations
python scripts/visualize_results.py --run-dir runs/my_first_run/

# Open the generated report
open runs/my_first_run/report.html
```

## 📖 Detailed Documentation

- [Configuration Guide](docs/CONFIG_GUIDE.md) - All configuration options explained
- [Architecture Overview](docs/ARCHITECTURE.md) - System design and interfaces
- [Extending the Platform](docs/EXTENDING.md) - Add new models, losses, metrics
- [API Reference](docs/API.md) - Detailed API documentation

## 🎯 Key Features

- **Pre-built Models**: U-Net, U-Net++, DeepLabV3+ with any timm encoder
- **Flexible Losses**: Dice, Tversky, BCE, Focal, and combinations
- **Comprehensive Metrics**: IoU, Dice, Precision, Recall, HD95
- **Data Augmentation**: Albumentations integration with 50+ transforms
- **Automatic Mixed Precision**: For faster training on GPUs
- **Cross-Validation**: K-fold CV with stratification and grouping
- **Test-Time Augmentation**: Improve predictions with TTA
- **Experiment Tracking**: Built-in logging, optional MLflow/W&B

## 🔧 Configuration System

All parameters controlled via YAML configs:

```yaml
model:
  architecture: "Unet"        # or "UnetPlusPlus", "DeepLabV3Plus"
  encoder: "efficientnet-b3"  # Any timm encoder
  encoder_weights: "imagenet"

training:
  epochs: 100
  batch_size: 16
  learning_rate: 1e-4
  
  loss:
    type: "dice"              # or "tversky", "bce", "focal"
    params:
      smooth: 1.0
```

[Full configuration guide →](docs/CONFIG_GUIDE.md)

## 📊 Supported Datasets

### Built-in Support
- DRIVE (retinal vessels)
- CHASE-DB1 (retinal vessels)  
- HRF (high-resolution fundus)
- STARE (retinal vessels)

### Custom Datasets
Simply point to your data:
```yaml
dataset:
  images_dir: "path/to/images"
  masks_dir: "path/to/masks"
```

[Dataset preparation guide →](docs/DATASETS.md)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Quick integration test
pytest tests/integration/test_training.py -k test_single_epoch
```

## 📈 Performance

| Dataset | Model | Encoder | Dice | IoU | Time (M1) |
|---------|-------|---------|------|-----|-----------|
| DRIVE | U-Net | ResNet34 | 0.82 | 0.76 | 12 min |
| DRIVE | U-Net++ | EfficientNet-B3 | 0.84 | 0.78 | 18 min |
| DRIVE | DeepLabV3+ | ResNet50 | 0.83 | 0.77 | 15 min |

## 🗺️ Roadmap

### Current Release (v2.0)
- ✅ Single train/val split training
- ✅ SMP model integration
- ✅ Lightning training loop
- ✅ Comprehensive metrics

### Next Release (v2.1)
- 🔄 K-fold cross-validation
- 🔄 Test-time augmentation
- 🔄 Sliding window inference

### Future (v3.0+)
- ⏳ MONAI integration
- ⏳ 3D segmentation support
- ⏳ Classification tasks
- ⏳ Multimodal fusion

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📝 Citation

If you use this platform in your research, please cite:
```bibtex
@software{segmentation_platform,
  title = {Image Segmentation Platform},
  author = {Your Team},
  year = {2025},
  url = {https://github.com/yourorg/segmentation-platform}
}
```

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.
```

## 22. Conclusion

This revised design document addresses the core issues while maintaining the modularity vision:

1. **Focused Scope**: Segmentation only, with clear upgrade path
2. **Leverages Libraries**: SMP + Lightning eliminates boilerplate
3. **Simple Start**: Single split MVP, k-fold immediately after
4. **Strong Documentation**: Comprehensive comments and guides required
5. **Clean Interfaces**: 5 stable contracts for long-term flexibility

The platform can be implemented in **2-3 weeks** for MVP, with k-fold CV and enhancements following immediately after validation.