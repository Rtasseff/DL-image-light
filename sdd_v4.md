# Image Segmentation Platform — Software Design Document
*Version:* 4.0  
*Date:* 2025-01-16  
*Status:* Production-Ready with Golden Path

## Executive Summary
Build a **production-ready** segmentation platform with a **clear golden path** and **optional safety nets**. Based on implementation experience through Week 2 and architectural review.

**Key Principles in v4.0:**
- **One Golden Path** - SMP + Lightning + Full dependencies + STRICT validation
- **Gated Complexity** - Fallbacks and auto-tuning are opt-in only
- **Explicit Over Magic** - All automatic behaviors must be logged
- **Test Without Block** - Three-tier testing enables continuous development
- **Version Pinning** - Exact versions for reproducibility

## 1. The Golden Path (START HERE)

### For 90% of Use Cases, Follow This Exact Path:

```bash
# Setup
pip install -r requirements/ml.txt  # Pinned versions
python scripts/train.py --config configs/base_config.yaml

# Configuration
validation_mode: STRICT  # Always for production
use_fallbacks: false     # Never in production
auto_tune: false         # Explicit control over experiments

# Platform
GPU (CUDA) or Apple Silicon (MPS)
Full ML dependencies installed
No fallback implementations
```

**If you deviate from this path, document why in your PR and config.**

### Golden Path Stack
- **Models**: Segmentation Models PyTorch (SMP) v0.3.3
- **Training**: PyTorch Lightning v2.1.2  
- **Metrics**: TorchMetrics v1.2.0
- **Augmentations**: Albumentations v1.3.1
- **Config Validation**: STRICT mode only
- **No Fallbacks**: Full dependencies required

## 2. Purpose & Scope

### Current Scope (MVP - Week 2 Complete)
- **Task:** Semantic segmentation
- **Models:** U-Net/U-Net++/DeepLabV3+ via SMP
- **Data:** Local PNG/JPEG images with mask labels
- **Validation:** Single train/val split with fixed seed
- **Target:** DRIVE dataset (retinal vessel segmentation)
- **Platform:** Golden Path = GPU/MPS, Fallback = CPU with explicit flag

### Immediate Next Phase (v4.1)
- **5-fold cross-validation** with external orchestration
- **Test-time augmentation** (opt-in only)
- **Sliding window inference** (opt-in for large images)

## 3. Core Technical Stack with Pinned Versions

### Requirements Structure
```
requirements/
├── base.txt        # Minimal core (always required)
├── ml.txt          # Golden path ML stack (pinned)
├── ml-loose.txt    # Same as ml.txt but with >= versions
├── dev.txt         # Development tools
└── all.txt         # Everything
```

### Golden Path Dependencies (requirements/ml.txt)
```txt
# Exact versions for reproducibility
torch==2.1.0
pytorch-lightning==2.1.2
segmentation-models-pytorch==0.3.3
torchmetrics==1.2.0
albumentations==1.3.1
timm==0.9.12
numpy==1.24.3
pillow==10.1.0
pyyaml==6.0.1
pydantic==2.5.0
tqdm==4.66.1
```

### Fallback Dependencies (requirements/base.txt)
```txt
# Minimal for testing only
torch>=2.0.0
numpy>=1.24.0
pillow>=10.0.0
pyyaml>=6.0
pydantic>=2.0.0
```

## 4. Architecture: Five Stable Interfaces

**These interfaces remain stable through v5.0. No changes without major version bump.**

### 4.1 DataModule Contract
```python
class DataModule(Protocol):
    """Stable API v4.0 - v5.0"""
    def setup(self, stage: str) -> None: ...
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> DataLoader: ...
    def test_dataloader(self) -> Optional[DataLoader]: ...
```

### 4.2 Model Contract  
```python
class Model(nn.Module):
    """Stable API v4.0 - v5.0"""
    def forward(self, x: Tensor) -> Tensor: ...
    def predict_step(self, x: Tensor, strategy: str = "standard") -> Tensor: ...
```

### 4.3 Loss Contract
```python
class Loss(Protocol):
    """Stable API v4.0 - v5.0"""
    def __call__(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor: ...
```

### 4.4 Metrics Contract
```python
class Metrics(Protocol):
    """Stable API v4.0 - v5.0"""
    def update(self, pred: Tensor, target: Tensor) -> None: ...
    def compute(self) -> Dict[str, Tensor]: ...
    def reset(self) -> None: ...
```

### 4.5 Trainer Contract
```python
class Trainer(Protocol):
    """Stable API v4.0 - v5.0"""
    def fit(self, model, datamodule, callbacks=None) -> None: ...
    def validate(self, model, datamodule) -> List[Dict]: ...
    def test(self, model, datamodule) -> List[Dict]: ...
```

## 5. Configuration Policy

### Validation Modes by Environment

| Environment | Mode | Behavior |
|------------|------|----------|
| **Local Development** | PERMISSIVE | Warnings on issues, fills defaults |
| **CI - Unit Tests** | MINIMAL | Only essential fields required |
| **CI - Integration** | STRICT | Full validation, no missing fields |
| **Production** | STRICT | Full validation, no missing fields |
| **Releases** | STRICT | Full validation, no missing fields |

### Mode Selection (Automatic)
```python
# src/core/config.py
def get_validation_mode() -> ValidationMode:
    """Auto-detect validation mode based on environment."""
    
    # Explicit override
    if mode := os.environ.get('CONFIG_VALIDATION_MODE'):
        return ValidationMode(mode)
    
    # CI environment
    if os.environ.get('CI'):
        if 'unit' in os.environ.get('TEST_SUITE', ''):
            return ValidationMode.MINIMAL
        return ValidationMode.STRICT
    
    # Local development
    if os.environ.get('USER'):  # Local machine
        return ValidationMode.PERMISSIVE
    
    # Default to strict
    return ValidationMode.STRICT
```

### Golden Path Configuration
```yaml
# configs/base_config.yaml - Golden Path defaults
project_name: "segmentation_experiment"
task: "segmentation"

# NO complexity flags in golden path
use_fallbacks: false  # NEVER true in production
validation_mode: null  # Auto-detected, not specified

dataset:
  images_dir: "./data/images"
  masks_dir: "./data/masks"
  split:
    type: "random"
    val_ratio: 0.2
    seed: 42

model:
  architecture: "Unet"
  encoder: "resnet34"
  encoder_weights: "imagenet"
  classes: 1

training:
  epochs: 50
  batch_size: 8  # Fixed, not auto-adjusted
  learning_rate: 1e-4
  optimizer: "adamw"
  
  loss:
    type: "dice"
    params:
      smooth: 1.0
  
  metrics: ["dice", "iou"]

# Resource management OFF by default
resources:
  auto_tune: false  # Must explicitly enable
  log_effective_settings: true  # Always log what ran
  
output:
  dir: "./runs"
  save_overlays: true
  checkpoint:
    monitor: "val_loss"
    mode: "min"
    save_best_only: true

# Simple visualization defaults
visualization:
  mode: "simple"  # Not "detailed" by default
  overlay_alpha: 0.3
  overlay_colormap: "viridis"

compute:
  accelerator: "auto"
  devices: 1
  precision: 32
  deterministic: true
  seed: 42
```

## 6. Fallback Policy

### When Fallbacks Are Allowed

| Context | Fallbacks Allowed | How to Enable |
|---------|-------------------|---------------|
| **Unit Tests** | ✅ Yes | Automatic in test environment |
| **Integration Tests (Minimal)** | ✅ Yes | Automatic in test environment |
| **Local Development** | ⚠️ Optional | `export USE_FALLBACKS=true` |
| **CI Full Tests** | ❌ No | Never |
| **Production** | ❌ No | Never |
| **Releases** | ❌ No | Never |

### Implementation
```python
# src/core/dependencies.py
import os
import warnings

def should_use_fallbacks() -> bool:
    """Determine if fallbacks should be used."""
    
    # Check explicit environment variable
    if os.environ.get('USE_FALLBACKS') == 'true':
        warnings.warn(
            "⚠️  Using fallback implementations. "
            "This should ONLY be used for testing or local development. "
            "Never deploy with fallbacks enabled.",
            UserWarning
        )
        return True
    
    # Auto-enable for specific test contexts
    if os.environ.get('PYTEST_CURRENT_TEST'):
        test_file = os.environ.get('PYTEST_CURRENT_TEST', '')
        if 'unit' in test_file or 'minimal' in test_file:
            return True
    
    return False

# Usage in imports
if should_use_fallbacks():
    from .testing.simple_metrics import SimpleDice as Dice
else:
    from torchmetrics import Dice  # Fail fast if missing
```

## 7. Effective Settings Logging

### Every Run Must Log What Actually Happened

```python
# src/core/effective_settings.py
"""Log effective settings that may differ from requested."""

from pathlib import Path
import json
from typing import Dict, Any
import warnings

class EffectiveSettingsLogger:
    """Track and log actual vs requested settings."""
    
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.requested = {}
        self.effective = {}
        self.changes = []
    
    def log_setting(
        self, 
        key: str, 
        requested: Any, 
        effective: Any, 
        reason: str = None
    ):
        """Log a setting that changed from requested."""
        self.requested[key] = requested
        self.effective[key] = effective
        
        if requested != effective:
            change = {
                'setting': key,
                'requested': requested,
                'effective': effective,
                'reason': reason or 'Unknown'
            }
            self.changes.append(change)
            
            # Warn user about change
            warnings.warn(
                f"Setting changed: {key} = {effective} "
                f"(requested: {requested}). Reason: {reason}",
                UserWarning
            )
    
    def save(self):
        """Save effective settings to file."""
        output = {
            'requested_settings': self.requested,
            'effective_settings': self.effective,
            'changes': self.changes,
            'change_count': len(self.changes),
            'fallbacks_used': self._detect_fallbacks(),
            'platform_optimizations': self._detect_optimizations()
        }
        
        output_path = self.run_dir / 'effective_settings.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        if self.changes:
            print(f"\n⚠️  {len(self.changes)} settings changed from requested values.")
            print(f"   See {output_path} for details.\n")
    
    def _detect_fallbacks(self) -> list:
        """Detect which fallback implementations are in use."""
        fallbacks = []
        
        # Check metrics
        from ..metrics import Dice
        if 'simple' in Dice.__module__:
            fallbacks.append('SimpleDice')
        
        # Check augmentations
        try:
            import albumentations
        except ImportError:
            fallbacks.append('BasicAugmentations')
        
        return fallbacks
    
    def _detect_optimizations(self) -> list:
        """Detect platform optimizations applied."""
        import torch
        optimizations = []
        
        if torch.cuda.is_available():
            optimizations.append('cuda_enabled')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            optimizations.append('mps_enabled')
        else:
            optimizations.append('cpu_only')
        
        return optimizations
```

### Example effective_settings.json
```json
{
  "requested_settings": {
    "batch_size": 16,
    "precision": 16,
    "accelerator": "auto"
  },
  "effective_settings": {
    "batch_size": 8,
    "precision": 32,
    "accelerator": "mps"
  },
  "changes": [
    {
      "setting": "batch_size",
      "requested": 16,
      "effective": 8,
      "reason": "Insufficient memory (3.2GB available, 6GB required)"
    },
    {
      "setting": "precision",
      "requested": 16,
      "effective": 32,
      "reason": "Mixed precision not supported on MPS"
    }
  ],
  "change_count": 2,
  "fallbacks_used": [],
  "platform_optimizations": ["mps_enabled"]
}
```

## 8. Auto-Tuning Policy (Opt-In Only)

### Disabled by Default, Explicit When Enabled

```yaml
# Only enable auto-tuning when explicitly requested
resources:
  auto_tune: false  # DEFAULT - no magic
  
  # When auto_tune: true, these apply:
  auto_batch_size:
    enabled: false  # Even with auto_tune, be explicit
    memory_fraction: 0.6  # Use 60% of available memory
    min_batch_size: 1
    max_batch_size: 32
  
  auto_precision:
    enabled: false
    prefer_mixed: true
```

### Auto-Tuning Implementation
```python
# src/core/auto_tune.py
"""Optional auto-tuning with explicit logging."""

class AutoTuner:
    """Auto-tune settings when explicitly requested."""
    
    def __init__(self, config: Dict, logger: EffectiveSettingsLogger):
        self.config = config
        self.logger = logger
        self.enabled = config.get('resources', {}).get('auto_tune', False)
    
    def tune_batch_size(self, requested: int) -> int:
        """Auto-tune batch size if enabled."""
        
        if not self.enabled:
            return requested
        
        if not self.config['resources'].get('auto_batch_size', {}).get('enabled', False):
            return requested
        
        # Calculate based on available memory
        from ..utils.memory import MemoryManager
        suggested = MemoryManager.suggest_batch_size()
        
        if suggested != requested:
            self.logger.log_setting(
                'batch_size',
                requested=requested,
                effective=suggested,
                reason=f"Auto-tuned based on available memory"
            )
        
        return suggested
```

## 9. Testing Strategy (Three Tiers)

### Test Organization
```
tests/
├── unit/              # No ML dependencies
│   ├── test_config.py
│   ├── test_io.py
│   └── test_registry.py
├── integration/
│   ├── minimal/       # Core dependencies only, fallbacks OK
│   │   └── test_pipeline_minimal.py
│   └── full/          # All dependencies required, no fallbacks
│       └── test_pipeline_full.py
└── smoke/             # Quick validation
    └── test_smoke.py
```

### Test Execution Policy

```bash
# For most development (always works)
pytest tests/unit

# For integration without full stack
USE_FALLBACKS=true pytest tests/integration/minimal

# For full validation (golden path)
pytest tests/integration/full  # Will fail if deps missing

# For CI
make test-ci  # Runs appropriate tier based on environment
```

### pytest Configuration
```python
# tests/conftest.py
import os
import pytest

def pytest_configure(config):
    """Configure pytest based on environment."""
    
    # Determine test tier
    if os.environ.get('CI'):
        if os.environ.get('TEST_TIER') == 'unit':
            os.environ['USE_FALLBACKS'] = 'true'
            os.environ['CONFIG_VALIDATION_MODE'] = 'MINIMAL'
        elif os.environ.get('TEST_TIER') == 'full':
            os.environ['USE_FALLBACKS'] = 'false'
            os.environ['CONFIG_VALIDATION_MODE'] = 'STRICT'
    
    # Mark slow tests
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
```

## 10. Repository Structure (Simplified)

```
segmentation-platform/
├── src/
│   ├── core/
│   │   ├── config.py            # Validation with auto-detection
│   │   ├── dependencies.py      # Fallback control
│   │   ├── effective_settings.py # Log what actually ran
│   │   ├── trainer.py           # Lightning wrapper
│   │   └── auto_tune.py         # Opt-in auto-tuning
│   │
│   ├── models/                  # Golden path models
│   │   ├── segmentation.py      # SMP models only
│   │   └── lightning_module.py
│   │
│   ├── testing/                 # Quarantined test helpers
│   │   ├── simple_metrics.py    # Fallback metrics
│   │   ├── simple_unet.py       # Fallback model
│   │   └── fixtures.py          # Test data
│   │
│   └── [other modules...]
│
├── configs/
│   ├── base_config.yaml         # Golden path config
│   ├── examples/
│   │   ├── low_memory.yaml      # auto_tune: true example
│   │   └── development.yaml     # use_fallbacks: true example
│   └── ci/
│       ├── strict.yaml          # CI config
│       └── minimal.yaml         # Test config
│
├── scripts/
│   ├── train.py                 # Golden path training
│   ├── predict.py              
│   ├── evaluate.py
│   └── check_dependencies.py    # Diagnostic tool
│
├── requirements/
│   ├── base.txt                # Core only
│   ├── ml.txt                  # Golden path (pinned)
│   ├── ml-loose.txt            # Same as ml.txt with >=
│   ├── dev.txt                 # Development tools
│   └── all.txt                 # Everything
│
├── docs/
│   ├── README.md               # Golden path quickstart
│   ├── TROUBLESHOOTING.md      # Fallbacks, auto-tuning
│   └── ADVANCED.md             # Platform optimizations
│
└── Makefile                    # Common commands
```

## 11. Documentation Structure

### README.md - Golden Path Only
```markdown
# Image Segmentation Platform

## Quick Start (Golden Path)

```bash
# Install dependencies (pinned versions)
pip install -r requirements/ml.txt

# Train model
python scripts/train.py --config configs/base_config.yaml

# Results in runs/<timestamp>/
```

## Requirements
- GPU with CUDA or Apple Silicon with MPS
- Python 3.10+
- 8GB+ RAM

That's it! For troubleshooting, see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).
```

### TROUBLESHOOTING.md - Complexity Lives Here
```markdown
# Troubleshooting

## Missing Dependencies
If you can't install the full ML stack:
```bash
# For testing only
export USE_FALLBACKS=true
pip install -r requirements/base.txt
python scripts/train.py --config configs/examples/development.yaml
```
⚠️ Never use fallbacks in production!

## Low Memory
For systems with <8GB RAM:
```yaml
# configs/examples/low_memory.yaml
resources:
  auto_tune: true
  auto_batch_size:
    enabled: true
```

## Platform Issues
[Platform-specific solutions...]
```

## 12. Dependency Diagnostic Tool

```python
# scripts/check_dependencies.py
"""Comprehensive dependency checker."""

import sys
import json
import importlib.util
from pathlib import Path
from typing import Dict, Tuple

def check_golden_path() -> bool:
    """Check if golden path dependencies are satisfied."""
    required = {
        'torch': '2.1.0',
        'pytorch_lightning': '2.1.2',
        'segmentation_models_pytorch': '0.3.3',
        'torchmetrics': '1.2.0',
        'albumentations': '1.3.1',
    }
    
    all_present = True
    print("🏆 Golden Path Dependencies:")
    print("-" * 50)
    
    for package, expected_version in required.items():
        actual_version = get_package_version(package)
        if actual_version == expected_version:
            print(f"  ✅ {package:30} {actual_version}")
        elif actual_version:
            print(f"  ⚠️  {package:30} {actual_version} (expected {expected_version})")
            all_present = False
        else:
            print(f"  ❌ {package:30} Not installed")
            all_present = False
    
    return all_present

def get_package_version(package: str) -> str:
    """Get installed package version."""
    try:
        spec = importlib.util.find_spec(package)
        if spec:
            module = importlib.import_module(package)
            return getattr(module, '__version__', 'Unknown')
    except:
        pass
    return None

def main():
    """Run diagnostic."""
    print("\n🔍 Dependency Diagnostic Report\n")
    print("=" * 50)
    
    # Check golden path
    golden_path_ok = check_golden_path()
    
    # Platform info
    print("\n💻 Platform Information:")
    print("-" * 50)
    import platform
    import torch
    
    print(f"  OS:          {platform.system()} {platform.release()}")
    print(f"  Python:      {platform.python_version()}")
    print(f"  PyTorch:     {torch.__version__}")
    
    if torch.cuda.is_available():
        print(f"  CUDA:        ✅ {torch.version.cuda}")
        print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    else:
        print(f"  CUDA:        ❌ Not available")
    
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print(f"  MPS:         ✅ Available")
    else:
        print(f"  MPS:         ❌ Not available")
    
    # Recommendations
    print("\n💡 Recommendations:")
    print("-" * 50)
    
    if golden_path_ok:
        print("  ✅ Golden path satisfied! You're ready for production.")
    else:
        print("  ⚠️  Golden path not satisfied.")
        print("     For production: pip install -r requirements/ml.txt")
        print("     For development: export USE_FALLBACKS=true")
    
    # Save report
    report = {
        'golden_path_satisfied': golden_path_ok,
        'timestamp': str(Path.cwd()),
        'can_use_fallbacks': not golden_path_ok,
    }
    
    with open('dependency_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Report saved to dependency_report.json")
    
    sys.exit(0 if golden_path_ok else 1)

if __name__ == "__main__":
    main()
```

## 13. Cross-Validation (External Orchestration)

### K-Fold Implementation (One Trainer Per Fold)
```python
# src/evaluation/cross_validation.py
"""External k-fold orchestration - no sneaky leakage."""

from pathlib import Path
import json
import numpy as np
from typing import List, Dict, Tuple

class CrossValidator:
    """
    External orchestration of k-fold CV.
    Each fold gets fresh model and trainer - no state leakage.
    """
    
    @staticmethod
    def run_kfold(
        config_path: Path,
        k: int = 5,
        output_dir: Path = Path("runs/cv")
    ) -> Tuple[List[Dict], Dict]:
        """
        Run k-fold cross-validation.
        
        Each fold is completely independent:
        - Fresh model initialization
        - Fresh trainer
        - Separate output directory
        """
        
        fold_results = []
        
        for fold in range(k):
            print(f"\n{'='*50}")
            print(f"Fold {fold + 1}/{k}")
            print(f"{'='*50}\n")
            
            # Fresh everything for this fold
            fold_dir = output_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            
            # Run training for this fold
            result = run_single_fold(
                config_path=config_path,
                fold_idx=fold,
                total_folds=k,
                output_dir=fold_dir
            )
            
            fold_results.append(result)
            
            # Save fold results immediately
            with open(fold_dir / "metrics.json", "w") as f:
                json.dump(result, f, indent=2)
        
        # Aggregate after all folds complete
        summary = aggregate_fold_results(fold_results)
        
        # Save summary
        with open(output_dir / "cv_summary.json", "w") as f:
            json.dump({
                'folds': fold_results,
                'summary': summary
            }, f, indent=2)
        
        print_cv_summary(summary)
        
        return fold_results, summary

def aggregate_fold_results(fold_results: List[Dict]) -> Dict:
    """Aggregate metrics across folds."""
    
    metrics = {}
    for metric_name in fold_results[0]['metrics'].keys():
        values = [f['metrics'][metric_name] for f in fold_results]
        metrics[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'ci95': 1.96 * np.std(values) / np.sqrt(len(values))
        }
    
    return metrics

def print_cv_summary(summary: Dict):
    """Print formatted CV results."""
    print("\n" + "="*50)
    print("Cross-Validation Summary")
    print("="*50)
    
    for metric, stats in summary.items():
        print(f"\n{metric}:")
        print(f"  Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        print(f"  95% CI: [{stats['mean']-stats['ci95']:.4f}, {stats['mean']+stats['ci95']:.4f}]")
        print(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
```

## 14. Makefile for Common Commands

```makefile
# Makefile - Common commands

.PHONY: help install test train clean

help:
	@echo "Usage:"
	@echo "  make install      Install golden path dependencies"
	@echo "  make install-dev  Install with dev tools"
	@echo "  make test        Run appropriate tests"
	@echo "  make test-all    Run all tests"
	@echo "  make train       Train with base config"
	@echo "  make clean       Clean runs and cache"

install:
	pip install -r requirements/ml.txt

install-dev:
	pip install -r requirements/all.txt

test:
	# Run appropriate tier based on environment
	@if [ -f ".venv/lib/python*/site-packages/torchmetrics" ]; then \
		echo "Running full tests..."; \
		pytest tests/; \
	else \
		echo "Running unit tests only..."; \
		USE_FALLBACKS=true pytest tests/unit tests/integration/minimal; \
	fi

test-all:
	pytest tests/

test-unit:
	USE_FALLBACKS=true pytest tests/unit

test-minimal:
	USE_FALLBACKS=true pytest tests/integration/minimal

test-full:
	pytest tests/integration/full

train:
	python scripts/train.py --config configs/base_config.yaml

check-deps:
	python scripts/check_dependencies.py

clean:
	rm -rf runs/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
```

## 15. Version History & Migration

### Version 4.0 Changes from v3.0
- **Golden Path Definition**: Clear primary workflow
- **Gated Complexity**: Fallbacks/auto-tuning off by default
- **Effective Settings Logging**: Track all deviations
- **Simplified Testing**: Three clear tiers
- **Pinned Dependencies**: Exact versions in ml.txt
- **External CV**: K-fold orchestration outside trainer

### Migration from v3.0
1. Update configs to set `use_fallbacks: false`
2. Update configs to set `auto_tune: false`  
3. Pin dependencies with `pip freeze > requirements/ml.txt`
4. Move fallback implementations to `src/testing/`
5. Add effective settings logging to training scripts

## 16. Conclusion

This Software Design Document v4.0 provides a **production-ready platform** with:

1. **Clear Golden Path** - One way that works for 90% of cases
2. **Gated Complexity** - Advanced features available but opt-in
3. **Explicit Behavior** - All automatic actions logged
4. **Flexible Testing** - Three tiers enable continuous development
5. **Reproducibility** - Pinned versions and effective settings tracking

The platform maintains power and flexibility while avoiding the complexity trap through explicit opt-in mechanisms and comprehensive logging of any deviations from requested behavior.

**Remember: When in doubt, follow the Golden Path.**