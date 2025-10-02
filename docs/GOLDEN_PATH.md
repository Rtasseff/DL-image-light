# The Golden Path - Complete Walkthrough

This guide walks you through a complete segmentation project from start to finish, explaining each step and common variations.

**Time to read**: 15 minutes  
**Level**: Beginner to intermediate

---

## Overview

You have images with some regions you want to identify (blood vessels, tumors, roads, etc.). This platform trains a neural network to automatically segment those regions in new images.

**What you'll learn**:
1. How to structure your data
2. How to configure training
3. How to evaluate results
4. How to use the trained model

---

## Step 1: Prepare Your Data

### Required Structure
```
data/
├── images/              # Your input images
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/               # Binary masks (white=target, black=background)
    ├── image_001_mask.png
    ├── image_002_mask.png
    └── ...
```

### Rules
- **Images**: Any format (PNG, JPG, TIFF). RGB or grayscale.
- **Masks**: Same dimensions as corresponding image
- **Binary masks**: 0 (black) = background, 255 (white) = target region
- **Naming**: Masks should match images (configurable but keep it simple)

### How Much Data?
- **Minimum**: 50 images (25 train, 25 validation)
- **Recommended**: 200+ images
- **More is better**: But 1000+ images means you should consider advanced splitting

### Example Mask Creation
If you have annotations from a tool (e.g., labelme, CVAT):
```python
# Convert polygons to binary mask
from PIL import Image, ImageDraw
import numpy as np

img = Image.open('image.png')
mask = Image.new('L', img.size, 0)
draw = ImageDraw.Draw(mask)
draw.polygon([(x1,y1), (x2,y2), ...], fill=255)
mask.save('mask.png')
```

---

## Step 2: Configure Training

### Start with Base Config
Copy `configs/base_config.yaml` and modify:

```yaml
# myproject_config.yaml
project_name: "my_segmentation_project"

dataset:
  images_dir: "./data/images"
  masks_dir: "./data/masks"
  split:
    val_ratio: 0.2        # 20% for validation
    seed: 42              # Reproducibility

model:
  architecture: "Unet"    # Start here, always
  encoder: "resnet34"     # Good balance speed/accuracy
  encoder_weights: "imagenet"

training:
  epochs: 50              # Usually enough
  batch_size: 8           # Adjust if memory issues
  learning_rate: 1e-4     # Default works well
  
  loss:
    type: "dice"          # Best for segmentation
    params:
      smooth: 1.0
  
  metrics: ["dice", "iou"]

output:
  dir: "./runs"
  save_overlays: true
  checkpoint:
    monitor: "val_loss"
    mode: "min"

compute:
  accelerator: "auto"     # Finds GPU/MPS automatically
  precision: 32           # Don't change unless you know why
  seed: 42
```

### Decision Points

#### Model Architecture (pick one)
| Architecture | Speed | Accuracy | Memory | When to Use |
|--------------|-------|----------|--------|-------------|
| **Unet** | Fast | Good | Low | **Start here - default choice** |
| UnetPlusPlus | Medium | Better | Medium | Need higher accuracy |
| DeepLabV3Plus | Slow | Best | High | Complex scenes, multiple scales |

**Recommendation**: Use Unet unless you have specific needs.

#### Encoder (pick one)
| Encoder | Parameters | Speed | When to Use |
|---------|------------|-------|-------------|
| resnet18 | 11M | Fastest | **Low memory systems** |
| **resnet34** | 21M | Fast | **Default - good balance** |
| resnet50 | 23M | Medium | Need more capacity |
| efficientnet-b0 | 4M | Fast | Mobile/edge deployment |

**Recommendation**: Use resnet34. Only change if memory issues (→ resnet18) or need more accuracy (→ resnet50).

#### Loss Function (pick one)
| Loss | When to Use | Params |
|------|-------------|--------|
| **dice** | **Balanced datasets, default** | smooth: 1.0 |
| focal | Imbalanced (small targets) | alpha: 0.25, gamma: 2.0 |
| tversky | Reduce false positives | alpha: 0.7, beta: 0.3 |
| focal_tversky | Very imbalanced + tuning | Multiple params |

**Recommendation**: Use dice. Only change if results show severe class imbalance.

---

## Step 3: Train the Model

### Basic Training
```bash
python scripts/train.py \
  --config myproject_config.yaml \
  --name experiment_001
```

### What Happens
1. Loads configuration and validates it
2. Splits data into train/validation sets
3. Creates model from config
4. Trains for specified epochs
5. Saves checkpoints in `runs/experiment_001/`

### Monitor Training
Watch the terminal output:
```
Epoch 10/50: 100%|██████| train_loss=0.234 val_loss=0.198 val_dice=0.856
```

**What to watch**:
- `train_loss` should decrease steadily
- `val_loss` should decrease (if increasing → overfitting)
- `val_dice` should increase (higher = better, max = 1.0)

### When to Stop
- **Validation loss stops improving** for 5-10 epochs → Stop early
- **Validation dice > 0.85** → Usually good enough
- **Train/val gap is large** → Need more data or regularization

### Common Issues

#### "Out of memory"
```yaml
# Solution 1: Reduce batch size
training:
  batch_size: 4  # or 2, or 1

# Solution 2: Smaller model
model:
  encoder: "resnet18"
```

#### "Training is very slow"
```yaml
# Use smaller encoder
model:
  encoder: "resnet18"

# Reduce epochs
training:
  epochs: 30
```

#### "Dice score is low (<0.7)"
1. **Check your masks first** - Most common issue
2. Try focal loss for imbalanced data
3. Train longer (100 epochs)
4. Check if images need different preprocessing

---

## Step 4: Evaluate Results

### Run Evaluation
```bash
python scripts/evaluate.py \
  --checkpoint runs/experiment_001/checkpoints/best.ckpt \
  --config runs/experiment_001/config.yaml \
  --output evaluation_001/
```

### What You Get
```
evaluation_001/
├── overall_metrics.json       # Summary stats
├── per_image_metrics.csv      # Detailed breakdown
├── metrics_summary.csv        # Statistics across dataset
├── confusion_matrix.png       # Visualization
└── visualizations/            # Side-by-side comparisons
    ├── error_analysis_000.png
    ├── error_analysis_001.png
    └── ...
```

### Interpreting Metrics

From `overall_metrics.json`:
```json
{
  "dice_score": 0.856,      // Overall accuracy (0-1, higher better)
  "iou": 0.748,             // Intersection over Union
  "precision": 0.891,       // How many predictions are correct
  "recall": 0.825,          // How much of target found
  "accuracy": 0.967         // Pixel-level accuracy
}
```

**What's Good?**
- Dice > 0.85: Excellent
- Dice 0.75-0.85: Good
- Dice 0.60-0.75: Acceptable (depends on use case)
- Dice < 0.60: Needs improvement

**Precision vs Recall**
- **High precision, low recall**: Model is cautious (misses some targets)
  - Solution: Increase false positive tolerance
- **Low precision, high recall**: Model is aggressive (too many false positives)
  - Solution: Increase threshold or use focal loss

### Visual Inspection
Open `visualizations/error_analysis_*.png`:
- **Green**: Correct predictions
- **Red**: False positives (model said yes, should be no)
- **Blue**: False negatives (model said no, should be yes)

Look for patterns:
- Errors at edges → Normal, acceptable
- Entire regions missed → Model needs improvement
- Random scattered errors → May need more data

---

## Step 5: Use the Model (Prediction)

### Predict on New Images
```bash
python scripts/predict.py \
  --checkpoint runs/experiment_001/checkpoints/best.ckpt \
  --config runs/experiment_001/config.yaml \
  --input new_images/ \
  --output predictions/
```

### Output Structure
```
predictions/
├── binary_masks/           # Black/white masks
│   ├── new_001_mask.png
│   └── ...
├── raw_predictions/        # Probability maps (0-1)
│   ├── new_001_prob.npy
│   └── ...
└── overlays/              # Visualizations
    ├── new_001_overlay.png
    └── ...
```

### Using Predictions Programmatically
```python
from PIL import Image
import numpy as np

# Load binary mask
mask = Image.open('predictions/binary_masks/image_mask.png')
mask_array = np.array(mask) > 127  # Boolean array

# Load probability map
probs = np.load('predictions/raw_predictions/image_prob.npy')
# probs is 0-1, higher = more confident

# Custom threshold
custom_mask = probs > 0.7  # More strict than default 0.5
```

---

## Step 6: Iterate and Improve

### Workflow for Better Results

```
1. Train with default config
   ↓
2. Evaluate on validation set
   ↓
3. Identify main issue:
   
   Low accuracy overall?
   → Try UnetPlusPlus, train longer
   
   Missing small objects?
   → Use focal loss
   
   Too many false positives?
   → Use tversky loss (alpha=0.7)
   
   Edges are rough?
   → This is normal, consider post-processing
   
   ↓
4. Adjust config, train again
   ↓
5. Compare experiments
   ↓
6. Use best checkpoint
```

### Comparing Experiments
```bash
# List all experiments
ls runs/

# Compare metrics
cat runs/experiment_001/overall_metrics.json
cat runs/experiment_002/overall_metrics.json

# Visual comparison
diff runs/experiment_001/config.yaml runs/experiment_002/config.yaml
```

---

## Advanced: Fine-Tuning

### Learning Rate Scheduling
```yaml
training:
  scheduler:
    type: "cosine"        # Gradually reduce LR
    min_lr: 1e-6
```

### Data Augmentation
```yaml
dataset:
  augmentation:
    enabled: true
    horizontal_flip_prob: 0.5
    vertical_flip_prob: 0.5
    rotation_limit: 15
    brightness_limit: 0.2
```

### Early Stopping
```yaml
training:
  early_stopping:
    monitor: "val_loss"
    patience: 10          # Stop if no improvement for 10 epochs
    mode: "min"
```

---

## Reproducibility Checklist

✅ **Save your config file** with the experiment  
✅ **Document your data source** (where did images come from?)  
✅ **Note the date** (data may change over time)  
✅ **Record the commit hash** (check `experiment_identity.json`)  
✅ **Keep validation set separate** (never train on it)  

The platform automatically tracks:
- Git commit hash
- All configuration used
- Environment (Python, CUDA versions)
- Random seeds

Check: `runs/experiment_001/experiment_identity.json`

---

## When This Guide Isn't Enough

### You need to:
- **Build custom architectures** → You need a different platform
- **Deploy in production** → See [ADVANCED/deployment.md]
- **Handle multi-class segmentation** → See [ADVANCED/multiclass.md]
- **Run cross-validation** → See [ADVANCED/cross_validation.md]
- **Customize data loading** → See [SDD.md Section 4.1]
- **Add new loss functions** → See [SDD.md Section 4.3]
- **Contribute code** → Read [SDD.md] completely

### You're hitting:
- **Weird configuration errors** → Check [CONTRACTS/validation_rules.md]
- **Import/dependency issues** → See [ADVANCED/fallback_system.md]
- **Multi-GPU training needs** → This is v2.0 feature
- **Real-time inference needs** → This platform is for training only

---

## Summary: The Happy Path

```bash
# 1. Setup (once)
pip install -r requirements/ml.txt

# 2. For each project:
#    - Prepare data in data/images and data/masks
#    - Copy and customize config

# 3. Train
python scripts/train.py --config my_config.yaml --name exp_001

# 4. Evaluate
python scripts/evaluate.py \
  --checkpoint runs/exp_001/checkpoints/best.ckpt \
  --config runs/exp_001/config.yaml \
  --output eval/

# 5. If good → Use it
python scripts/predict.py \
  --checkpoint runs/exp_001/checkpoints/best.ckpt \
  --config runs/exp_001/config.yaml \
  --input new_data/ \
  --output results/

# 6. If not good → Adjust config, go to step 3
```

**You're now ready to segment images!**
