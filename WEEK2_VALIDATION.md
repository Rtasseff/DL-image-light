# Week 2 Full MVP - Validation Instructions

## Prerequisites
```bash
# Ensure you're in the project directory and virtual environment is activated
cd DL-image-light
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install all dependencies (this will take a few minutes)
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Phase 1: Core Functionality Verification

### 1.1 Test Loss Functions
```bash
# Test all loss function imports and instantiation
python -c "
import sys
sys.path.append('src')
from src.losses import get_loss

# Test all Week 2 loss functions
losses = ['dice', 'tversky', 'focal', 'focal_tversky', 'asymmetric_focal']
for loss_name in losses:
    loss_fn = get_loss(loss_name)
    print(f'✅ {loss_name} loss: {type(loss_fn).__name__}')

print('✅ All loss functions working correctly!')
"
```

### 1.2 Test Visualization Module
```bash
# Test visualization imports
python -c "
import sys
sys.path.append('src')
from src.visualization.overlays import create_overlay, apply_colormap
from src.visualization.metrics_plot import plot_training_curves
print('✅ Visualization module imports successfully')
"
```

### 1.3 Test Reporting Module
```bash
# Test reporting imports
python -c "
import sys
sys.path.append('src')
from src.reporting.report import generate_training_report, generate_evaluation_report
print('✅ Reporting module imports successfully')
"
```

## Phase 2: Script Functionality Tests

### 2.1 Training Script with New Loss Functions
```bash
# Test training with Focal loss (fast dev run)
python scripts/train.py \
  --config configs/base_config.yaml \
  --name focal_test \
  --fast-dev-run \
  --loss-type focal

# Test training with Focal-Tversky loss
python scripts/train.py \
  --config configs/base_config.yaml \
  --name focal_tversky_test \
  --fast-dev-run \
  --loss-type focal_tversky

# Test training with Asymmetric Focal loss
python scripts/train.py \
  --config configs/base_config.yaml \
  --name asymmetric_focal_test \
  --fast-dev-run \
  --loss-type asymmetric_focal
```

### 2.2 Prediction Script Test
```bash
# Create a dummy test image
python -c "
import numpy as np
from PIL import Image
import os

# Create a test image
test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
os.makedirs('test_data', exist_ok=True)
Image.fromarray(test_img).save('test_data/test_image.jpg')
print('✅ Test image created')
"

# Test prediction script help
python scripts/predict.py --help

# Test prediction on test image (requires trained model)
# This will fail if no model exists, but should show proper error handling
python scripts/predict.py \
  --checkpoint runs/focal_test/checkpoints/best.ckpt \
  --config runs/focal_test/config.yaml \
  --input test_data/test_image.jpg \
  --output test_predictions/ \
  2>&1 | head -10
```

### 2.3 Evaluation Script Test
```bash
# Test evaluation script help
python scripts/evaluate.py --help

# The actual evaluation requires a trained model and dataset
echo "✅ Evaluation script syntax verified"
```

### 2.4 Visualization Script Test
```bash
# Test visualization script help
python scripts/visualize_results.py --help

# Test on training run directory (if exists)
python scripts/visualize_results.py \
  --run-dir runs/focal_test \
  --summary \
  2>&1 | head -10
```

## Phase 3: End-to-End Pipeline Test

### 3.1 Complete Pipeline Run
```bash
# Run a complete short training session with visualization
python scripts/train.py \
  --config configs/base_config.yaml \
  --name week2_validation \
  --epochs 2 \
  --loss-type focal_tversky

# Generate visualizations for the run
python scripts/visualize_results.py \
  --run-dir runs/week2_validation \
  --summary

# Create a comprehensive report (if training completed)
python -c "
import sys
sys.path.append('src')
from pathlib import Path
from src.reporting.report import generate_training_report

run_dir = Path('runs/week2_validation')
if run_dir.exists():
    report_path = generate_training_report(run_dir)
    print(f'✅ Training report generated: {report_path}')
else:
    print('⚠️  No training run found, skipping report generation')
"
```

## Phase 4: Feature Completeness Check

### 4.1 Verify All Week 2 Components
```bash
# Check all required Week 2 files exist
python -c "
from pathlib import Path

required_files = [
    'src/losses/focal.py',
    'src/visualization/overlays.py',
    'src/visualization/metrics_plot.py',
    'src/reporting/report.py',
    'scripts/predict.py',
    'scripts/evaluate.py',
    'scripts/visualize_results.py'
]

missing = []
for file_path in required_files:
    if not Path(file_path).exists():
        missing.append(file_path)

if missing:
    print('❌ Missing files:')
    for f in missing:
        print(f'  - {f}')
else:
    print('✅ All Week 2 files present')

# Check focal loss classes
import sys
sys.path.append('src')
from src.losses.focal import FocalLoss, BinaryFocalLoss, FocalTverskyLoss, AsymmetricFocalLoss
print('✅ All focal loss variants available')
"
```

### 4.2 Verify Updated Documentation
```bash
# Check if README contains Week 2 features
grep -q "Focal-Tversky" README.md && echo "✅ README updated with Week 2 features" || echo "❌ README missing Week 2 info"
grep -q "AsymmetricFocalLoss" README.md && echo "✅ README contains asymmetric focal loss" || echo "❌ README missing asymmetric focal loss"
grep -q "visualization" README.md && echo "✅ README mentions visualization" || echo "❌ README missing visualization info"
```

## Expected Results

After running all validation steps, you should see:

1. **✅ All loss functions working correctly!**
2. **✅ Visualization module imports successfully**
3. **✅ Reporting module imports successfully**
4. **✅ All Week 2 files present**
5. **✅ All focal loss variants available**
6. **✅ README updated with Week 2 features**

If any step fails, it indicates an incomplete or incorrect implementation.

## Success Criteria

Week 2 Full MVP is considered complete when:

- [x] Three focal loss variants implemented (Focal, Focal-Tversky, Asymmetric Focal)
- [x] Visualization module with overlay generation and plotting
- [x] Complete prediction pipeline with batch processing
- [x] Comprehensive evaluation system with metrics
- [x] Results visualization automation
- [x] Professional HTML report generation
- [x] All scripts have proper help and error handling
- [x] Documentation updated with new features

## Troubleshooting

### Common Issues:

1. **Import errors**: Ensure you've run `pip install -r requirements.txt`
2. **Module not found**: Run `pip install -e .` to install the package
3. **Missing checkpoints**: Train a model first with the training script
4. **Permission errors**: Ensure you have write permissions in the project directory

### Dependencies Check:
```bash
pip list | grep -E "(torch|lightning|albumentations|matplotlib|pandas|jinja2)"
```

All listed packages should be installed for full functionality.