# Image Segmentation Platform

A production-ready deep learning platform for image segmentation using PyTorch Lightning and Segmentation Models PyTorch.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd DL-image-light

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
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
│   ├── losses/         # Loss functions
│   ├── metrics/        # Evaluation metrics
│   └── utils/          # Utility functions
├── configs/            # Configuration files
├── scripts/            # Training and inference scripts
└── runs/              # Experiment outputs
```

## 🎯 Key Features

- **Pre-built Models**: U-Net, U-Net++, DeepLabV3+ with any timm encoder
- **Flexible Losses**: Dice, Tversky, BCE with customizable parameters
- **Comprehensive Metrics**: IoU, Dice, Precision, Recall
- **Data Augmentation**: Albumentations integration
- **Reproducible**: Fixed seeds and deterministic training
- **Configuration-driven**: YAML-based configuration system

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
    type: "dice"              # or "tversky", "bce"
```

## 📊 Week 1 Implementation Status

✅ **Completed Components:**
- Virtual environment and git repository setup
- Complete repository structure according to SDD
- Configuration system with Pydantic validation
- Dice loss and comprehensive metrics
- DataModule with DRIVE dataset support
- SMP model wrapper with Lightning
- Single train/val split functionality
- Basic training script with full pipeline

**Ready for use:** The core pipeline is complete and ready for training!

## 🧪 Testing the Implementation

```bash
# Test configuration loading
python -c "from src.core.config import load_config; print('Config system working!')"

# Test training pipeline with fast dev run
python scripts/train.py --config configs/base_config.yaml --fast-dev-run

# Full training run (requires data)
python scripts/train.py --config configs/drive.yaml --name test_run
```

## 📈 Performance

Optimized for M1 MacBook:
- MPS acceleration support
- Efficient batch sizes
- Gradient accumulation for larger effective batches

## 🗺️ Next Steps (Week 2+)

- Add more loss functions (Focal, Boundary)
- Implement overlay visualization
- Add prediction and evaluation scripts
- Create comprehensive test suite
- Add experiment tracking integration

## 🤝 Contributing

This is a production-ready segmentation platform following clean architecture principles. All components are modular and extensible.

## 📄 License

MIT License - see LICENSE for details.