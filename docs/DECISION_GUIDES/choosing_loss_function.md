# Decision Guide: Choosing a Loss Function

**Context**: You're configuring `training.loss.type` in your YAML config.

**Time to read**: 5 minutes

---

## Quick Decision Tree

```
Start here
   ↓
Is your target region < 10% of image?
   ├─ YES → Use "focal" loss
   └─ NO → Continue
       ↓
   Do you have roughly equal foreground/background?
       ├─ YES → Use "dice" loss (STOP - you're done)
       └─ NO → Continue
           ↓
       Do you care more about missing targets (false negatives)?
           ├─ YES → Use "tversky" with alpha=0.3, beta=0.7
           └─ NO (care more about false positives)
               └─ Use "tversky" with alpha=0.7, beta=0.3
```

---

## The Loss Functions Explained

### 1. Dice Loss (Default)
**Use when**: Balanced datasets, general purpose

**Config**:
```yaml
training:
  loss:
    type: "dice"
    params:
      smooth: 1.0
```

**Pros**:
- Works well for most cases
- Fast to compute
- Well-understood
- Directly optimizes the metric you care about (Dice score)

**Cons**:
- Can struggle with very small objects
- Not ideal for imbalanced classes

**Example use cases**:
- Medical organ segmentation (liver, heart)
- Road extraction from satellite images
- General object segmentation

---

### 2. Focal Loss
**Use when**: Small targets, severe class imbalance

**Config**:
```yaml
training:
  loss:
    type: "focal"
    params:
      alpha: 0.25      # Weight for positive class
      gamma: 2.0       # Focus on hard examples
```

**Pros**:
- Excellent for small objects
- Handles extreme imbalance (target is <5% of image)
- Focuses learning on hard examples

**Cons**:
- More hyperparameters to tune
- Can be unstable early in training
- Slightly slower than Dice

**When to adjust params**:
- Target is VERY rare (< 1%): Increase `alpha` to 0.5-0.75
- Target is moderately rare (1-10%): Keep default `alpha=0.25`
- Training is unstable: Reduce `gamma` to 1.5

**Example use cases**:
- Tumor detection (small lesions)
- Blood vessel segmentation (thin structures)
- Defect detection in manufacturing

---

### 3. Tversky Loss
**Use when**: You want to control false positive vs false negative tradeoff

**Config**:
```yaml
training:
  loss:
    type: "tversky"
    params:
      alpha: 0.7       # False positive weight
      beta: 0.3        # False negative weight
      smooth: 1.0
```

**How to set alpha/beta**:
- `alpha + beta = 1.0` (they must sum to 1)
- **Higher alpha** (e.g., 0.7): Penalize false positives more → Model is conservative
- **Higher beta** (e.g., 0.7): Penalize false negatives more → Model is aggressive

**Use cases by setting**:
```yaml
# Conservative (avoid false alarms)
alpha: 0.7, beta: 0.3
# Use for: Medical diagnostics where false positives are costly

# Aggressive (find everything)
alpha: 0.3, beta: 0.7
# Use for: Screening where missing a case is critical

# Balanced (similar to Dice)
alpha: 0.5, beta: 0.5
# Use for: Just use Dice instead, it's simpler
```

**Example use cases**:
- Medical screening (high beta - don't miss anything)
- Autonomous driving (high alpha - don't hallucinate obstacles)
- Quality control (depends on cost of false alarms vs defects)

---

### 4. Focal-Tversky Loss
**Use when**: Combining benefits of both focal and tversky

**Config**:
```yaml
training:
  loss:
    type: "focal_tversky"
    params:
      alpha: 0.7
      beta: 0.3
      gamma: 1.33      # Focal parameter
      smooth: 1.0
```

**Pros**:
- Handles imbalance (like focal)
- Controls FP/FN tradeoff (like tversky)
- Best of both worlds

**Cons**:
- Most parameters to tune
- Can be tricky to get right
- Slower convergence sometimes

**When to use**:
- You tried focal and tversky separately
- Neither worked perfectly
- You have time to experiment

**Typical values**:
- `gamma=1.33`: Standard (from paper)
- `gamma=2.0`: More focus on hard examples
- `gamma=1.0`: Less aggressive (closer to tversky)

---

## Practical Workflow

### Step 1: Start Simple
```yaml
training:
  loss:
    type: "dice"
```
Train for 50 epochs. Evaluate.

### Step 2: Diagnose Issues

Check `evaluation_001/overall_metrics.json`:

```json
{
  "dice_score": 0.65,
  "precision": 0.45,    // Low! Too many false positives
  "recall": 0.92        // High! Finding most targets
}
```

**Low precision + High recall** → Too aggressive  
**Solution**: Use tversky with `alpha: 0.7, beta: 0.3`

```json
{
  "dice_score": 0.68,
  "precision": 0.88,    // High! Few false positives
  "recall": 0.55        // Low! Missing many targets
}
```

**High precision + Low recall** → Too conservative  
**Solution**: Use tversky with `alpha: 0.3, beta: 0.7`

```json
{
  "dice_score": 0.58,
  "precision": 0.62,
  "recall": 0.61
  // Note: Target is only 2% of image
}
```

**Both low + Imbalanced data** → Wrong loss  
**Solution**: Use focal with `alpha: 0.25, gamma: 2.0`

### Step 3: Fine-Tune

After changing loss, train again:
```bash
python scripts/train.py \
  --config experiment_002_focal.yaml \
  --name exp_002_focal
```

Compare results:
```bash
# Old (Dice)
cat runs/exp_001/overall_metrics.json

# New (Focal)
cat runs/exp_002_focal/overall_metrics.json
```

---

## Common Mistakes

### ❌ Changing loss function every few epochs
Train to completion before switching.

### ❌ Using focal for balanced data
It's overkill. Stick with dice.

### ❌ Setting alpha/beta randomly
Use the decision tree above or understand the tradeoff.

### ❌ Not checking class balance first
```python
# Check your data balance
import numpy as np
from pathlib import Path
from PIL import Image

total_pixels = 0
foreground_pixels = 0

for mask_path in Path('data/masks').glob('*.png'):
    mask = np.array(Image.open(mask_path))
    total_pixels += mask.size
    foreground_pixels += (mask > 127).sum()

ratio = foreground_pixels / total_pixels
print(f"Foreground: {ratio*100:.2f}%")

# If ratio < 10% → Consider focal loss
# If 10-40% → Dice is fine
# If 40-60% → Perfectly balanced, use dice
```

---

## Summary Table

| Loss | Best For | Key Param | Training Time | Complexity |
|------|----------|-----------|---------------|------------|
| **dice** | Balanced data, default | smooth=1.0 | Fast | ⭐ Simple |
| **focal** | Imbalanced (< 10%) | gamma=2.0 | Medium | ⭐⭐ Medium |
| **tversky** | Control FP/FN tradeoff | alpha, beta | Fast | ⭐⭐ Medium |
| **focal_tversky** | Complex cases | alpha, beta, gamma | Slower | ⭐⭐⭐ Complex |

---

## Still Not Sure?

1. **Start with dice** - It works 80% of the time
2. **Check class balance** - If < 10%, use focal
3. **Evaluate results** - Look at precision/recall
4. **Adjust if needed** - Use the diagnosis guide above

**Need help?** Post in discussions with:
- Your class balance percentage
- Current precision/recall scores
- Use case description

---

## References

- Dice Loss: Milletari et al., "V-Net" (2016)
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection" (2017)
- Tversky Loss: Salehi et al., "Tversky Loss for Image Segmentation" (2018)
- Implementation: See `src/losses/` for actual code

**Next**: [Choosing Architecture](choosing_architecture.md)
