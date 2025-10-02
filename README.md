# Image Segmentation Platform

A production-ready deep learning platform for image segmentation using PyTorch Lightning and Segmentation Models PyTorch.

## 📚 Documentation

- [docs/README_STREAMLINED.md](docs/README_STREAMLINED.md) — user quick start and troubleshooting
- [docs/GOLDEN_PATH.md](docs/GOLDEN_PATH.md) — full workflow walkthrough
- [docs/DECISION_GUIDES/](docs/DECISION_GUIDES/) — 5-minute decision helpers
- [docs/FOR_ENGINEERS.md](docs/FOR_ENGINEERS.md) — engineering onboarding and SDD usage
- [docs/CONTRACTS/](docs/CONTRACTS/) — frozen interfaces and validation policies

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd DL-image-light

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Golden Path dependencies (pinned versions for reproducibility)
pip install -r requirements/ml.txt
```

### Train Your First Model

```bash
# Train U-Net on DRIVE dataset (when data is available)
python scripts/train.py --config configs/drive.yaml --name my_first_run

# For testing with dummy data
python scripts/train.py --config configs/base_config.yaml --fast-dev-run
```

## 📁 Project Structure

```
DL-image-light/
├── src/
│   ├── core/           # Configuration and core utilities
│   ├── data/           # Data loading and transforms
│   ├── models/         # Lightning model implementations
│   ├── losses/         # Loss functions (Dice, Tversky, Focal)
│   ├── metrics/        # Evaluation metrics
│   ├── visualization/  # Plotting and overlay generation
│   ├── reporting/      # HTML report generation
│   └── utils/          # Utility functions
├── configs/            # Configuration files
├── scripts/            # Training, evaluation, and prediction scripts
├── tests/              # Comprehensive test suite
└── runs/              # Experiment outputs
```

## 🎯 Key Features

### Core Components
- **Pre-built Models**: U-Net, U-Net++, DeepLabV3+ with any timm encoder
- **Flexible Losses**: Dice, Tversky, Focal Loss (3 variants) with customizable parameters
- **Comprehensive Metrics**: IoU, Dice, Precision, Recall, F1-Score
- **Data Augmentation**: Albumentations integration with optimized transforms
- **Reproducible**: Fixed seeds and deterministic training
- **Configuration-driven**: YAML-based configuration system with Pydantic validation

### Advanced Features
- **Visualization Suite**: Training curves, overlay generation, error analysis
- **Professional Reporting**: HTML report generation with embedded images
- **Complete Pipeline**: Training, evaluation, prediction, and visualization scripts
- **Comprehensive Testing**: 36+ unit tests with 94% configuration coverage
- **Multi-platform Support**: CUDA, MPS (Apple Silicon), and CPU acceleration

## 🔧 Configuration

All parameters are controlled via YAML configs:

```yaml
model:
  architecture: "Unet"        # or "UnetPlusPlus", "DeepLabV3Plus"
  encoder: "resnet34"         # Any timm encoder

training:
  epochs: 50
  batch_size: 8
  learning_rate: 1e-4
  loss:
    type: "dice"              # or "tversky", "focal", "focal_tversky", "asymmetric_focal"

dataset:
  name: "drive"               # Built-in DRIVE dataset support
  augmentation:
    enabled: true
    horizontal_flip_prob: 0.5
    vertical_flip_prob: 0.5
```

## 📊 Implementation Status

### ✅ Week 1: Core Pipeline (Complete)
- Virtual environment and git repository setup
- Complete repository structure according to SDD
- Configuration system with Pydantic validation
- Dice loss and comprehensive metrics
- DataModule with DRIVE dataset support
- SMP model wrapper with Lightning
- Single train/val split functionality
- Basic training script with full pipeline

### ✅ Week 2: Full MVP (Complete)
- **Loss Functions**: Focal, Focal-Tversky, Asymmetric Focal Loss
- **Visualization Module**: Overlay generation, training curves, error analysis
- **Prediction Pipeline**: Complete inference script with batch processing
- **Evaluation System**: Comprehensive model assessment with per-image metrics
- **Results Visualization**: Automated plot and overlay generation
- **Professional Reporting**: HTML report generation with embedded visualizations

**Status:** Production-ready segmentation platform with complete pipeline!

## 🧪 Usage Examples

### Training a Model
```bash
# Quick test with dummy data
python scripts/train.py --config configs/base_config.yaml --fast-dev-run

# Full training run (requires DRIVE dataset)
python scripts/train.py --config configs/drive.yaml --name my_experiment

# Training with custom loss function
python scripts/train.py --config configs/focal_config.yaml --name focal_experiment
```

### Making Predictions
```bash
# Predict on single image
python scripts/predict.py \
  --checkpoint runs/my_experiment/checkpoints/best.ckpt \
  --config runs/my_experiment/config.yaml \
  --input path/to/image.jpg \
  --output predictions/

# Batch prediction on directory
python scripts/predict.py \
  --checkpoint runs/my_experiment/checkpoints/best.ckpt \
  --config runs/my_experiment/config.yaml \
  --input path/to/images/ \
  --output predictions/
```

### Model Evaluation
```bash
# Evaluate trained model
python scripts/evaluate.py \
  --checkpoint runs/my_experiment/checkpoints/best.ckpt \
  --config runs/my_experiment/config.yaml \
  --output evaluation_results/

# Generate comprehensive visualizations
python scripts/visualize_results.py \
  --run-dir runs/my_experiment \
  --output visualizations/
```

### Running Tests
```bash
# Run full test suite
python -m pytest tests/ -v

# Test specific module
python -m pytest tests/test_config.py -v
```

## 📈 Performance

Optimized for M1 MacBook:
- MPS acceleration support
- Efficient batch sizes
- Gradient accumulation for larger effective batches

## 🗺️ Future Enhancements (v2.1+)

- **Cross-validation Support**: Multi-fold validation implementation
- **Advanced Augmentations**: CutMix, MixUp, and domain-specific transforms
- **Model Ensembling**: Multiple model combination strategies
- **Experiment Tracking**: MLflow, Weights & Biases integration
- **Advanced Architectures**: Vision Transformers, Swin-Unet support
- **Multi-class Segmentation**: Extension beyond binary segmentation
- **Real-time Inference**: Optimized deployment pipeline

## 🔧 Troubleshooting

### Installation Issues

If you cannot install the full ML stack (`requirements/ml.txt`):

```bash
# For development/testing only (not for production)
export USE_FALLBACKS=true
export CONFIG_VALIDATION_MODE=PERMISSIVE
pip install -r requirements/base.txt
```

⚠️ **Warning**: Never use fallbacks in production!

### Other Installation Options

```bash
# Development with additional tools
pip install -r requirements/dev.txt

# All dependencies (includes everything)
pip install -r requirements/all.txt

# Check what's available
python scripts/check_dependencies.py
```

### SDD v4.1 Compliance

This project follows SDD v4.1 standards. For development:

```bash
# Enable permissive mode for development
export CONFIG_VALIDATION_MODE=PERMISSIVE

# Production always uses STRICT (default)
unset CONFIG_VALIDATION_MODE
```

## 🤝 Contributing

This is a production-ready segmentation platform following SDD v4.1 clean architecture principles. All components are modular and extensible.

## 📄 License

MIT License - see LICENSE for details.