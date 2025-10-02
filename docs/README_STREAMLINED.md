# Image Segmentation Platform - Quick Start

**Purpose**: Train deep learning models to segment images (identify regions/objects).

**For**: Data analysts who need to segment medical images, satellite imagery, or similar tasks.

**Not For**: Building custom architectures, deploying production APIs, or real-time inference.

---

## The 5-Minute Start

```bash
# 1. Install
pip install -r requirements/ml.txt

# 2. Prepare your data
#    - Put images in: data/images/
#    - Put masks in:  data/masks/

# 3. Train
python scripts/train.py --config configs/base_config.yaml --name my_experiment

# 4. Evaluate
python scripts/evaluate.py \
  --checkpoint runs/my_experiment/checkpoints/best.ckpt \
  --config runs/my_experiment/config.yaml \
  --output evaluation/

# 5. Predict on new images
python scripts/predict.py \
  --checkpoint runs/my_experiment/checkpoints/best.ckpt \
  --config runs/my_experiment/config.yaml \
  --input new_images/ \
  --output predictions/
```

**Done.** That's it for 90% of use cases.

---

## When Something Goes Wrong

| Problem | Solution | Details |
|---------|----------|---------|
| Out of memory | Reduce `batch_size` in config | See [debugging_memory.md](DECISION_GUIDES/debugging_memory.md) |
| Training too slow | Use smaller `encoder` (e.g., resnet18) | See [choosing_architecture.md](DECISION_GUIDES/choosing_architecture.md) |
| Poor accuracy | Check data quality first | See [debugging_performance.md](DECISION_GUIDES/debugging_performance.md) |
| Missing dependencies | `pip install -r requirements/ml.txt` | See [installation.md](ADVANCED/installation.md) |
| Something else | Read [GOLDEN_PATH.md](GOLDEN_PATH.md) | Full walkthrough with explanations |

---

## Configuration Presets (Just Copy These)

### Standard Use Case (Medical Images)
```yaml
# configs/base_config.yaml
model:
  architecture: "Unet"
  encoder: "resnet34"
  
training:
  epochs: 50
  batch_size: 8
  learning_rate: 1e-4
  loss:
    type: "dice"
```

### Low Memory System (<8GB RAM)
```yaml
model:
  encoder: "resnet18"  # Smaller model
  
training:
  batch_size: 4  # Smaller batches
```

### High Accuracy Needed
```yaml
model:
  architecture: "UnetPlusPlus"
  encoder: "resnet50"
  
training:
  epochs: 100
  loss:
    type: "focal_tversky"  # Handles class imbalance
```

---

## The Rules (Don't Break These)

1. **Always use configs from `configs/` directory** - Don't edit code
2. **Never use fallbacks in real work** - Only for testing
3. **One experiment = one config file** - Helps reproducibility
4. **Check git status before training** - Avoid "uncommitted changes" warnings
5. **Save your configs** - You'll forget what you did

---

## For Developers Building This Platform

You need the full specification: [SDD.md](../sdd.md)

That document has:
- Architecture contracts that must remain stable
- Validation rules and why they exist
- Testing strategy for contributors
- Extension points for new features

**Engineers**: Treat SDD.md as the source of truth. Treat this README as the user manual.


## Stuck or Found Something New?

- Log reproducible steps in `docs/MANUAL_TESTS.md` before trying fixes.
- Add the workaround to this table or create a short decision guide (see `DECISION_GUIDES/`); link it back to the manual test entry.
- If a change requires new behavior, capture the goal and impacted areas in `docs/ACTION_PLAN.md` so engineers can scope work.

---

## Need More Detail?

- **I want to understand the workflow**: → [GOLDEN_PATH.md](GOLDEN_PATH.md)
- **I need to choose architecture/loss**: → [DECISION_GUIDES/](DECISION_GUIDES/)
- **I'm a developer/contributor**: → [SDD.md](../sdd.md)
- **I have weird requirements**: → [ADVANCED/](ADVANCED/)
