# Decision Guide: Choosing an Architecture

**Context**: You're setting `model.architecture` in your YAML config.

**Time to read**: 5 minutes

---

## Quick Decision Flow

```
Start with Unet (default)
   ↓
Are you missing fine detail or thin structures?
   ├─ YES → Try "UnetPlusPlus"
   └─ NO → Continue
       ↓
Are objects multi-scale or scenes very complex?
       ├─ YES → Consider "DeepLabV3Plus"
       └─ NO → Stay with "Unet"
```

---

## Architecture Comparison

| Architecture | Training Speed | Accuracy | Memory Use | Best When |
|--------------|----------------|----------|------------|-----------|
| **Unet** | Fast | Strong baseline | Low | General segmentation, first runs |
| UnetPlusPlus | Medium | Higher on boundaries | Medium | Fine structures, thin vessels |
| DeepLabV3Plus | Slowest | Best for complex scenes | High | Large context, multi-scale objects |

---

## Recommendations

- **Start with Unet** for every new project; it is the Golden Path default and easiest to tune.
- Switch to **UnetPlusPlus** when prediction masks look jagged, miss thin structures, or you need smoother boundaries without redesigning the pipeline.
- Move to **DeepLabV3Plus** only after data quality checks; it increases GPU memory needs and training time but handles multi-scale features better.
- When changing architecture, keep other settings (encoder, loss, batch size) identical for one experiment so you can attribute differences correctly.

---

## Config Snippets

```yaml
# Default baseline
model:
  architecture: "Unet"
  encoder: "resnet34"
```

```yaml
# Focus on boundary detail
model:
  architecture: "UnetPlusPlus"
  encoder: "resnet34"
```

```yaml
# Complex, multi-scale scenes
model:
  architecture: "DeepLabV3Plus"
  encoder: "resnet50"
```

---

## Metrics to Watch After Changes

- **Dice / IoU** should improve by ≥0.02 when moving off Unet; otherwise revert.
- **GPU memory** usage increases with UnetPlusPlus (+10-20%) and DeepLabV3Plus (+30-40%); reduce `training.batch_size` if you see OOM.
- **Training epochs** may need to increase for DeepLabV3Plus (e.g., 70-100) because convergence is slower.

---

## When to Escalate

- You hit OOM even after lowering batch size → switch back to Unet or use a smaller encoder (see `choosing_encoder.md`).
- Accuracy regresses across architectures → re-check data preprocessing or consider different loss (see `choosing_loss_function.md`).
- Stakeholders need multi-class segmentation → this platform is single-class; raise with engineering leads before modifying architecture.

---

## Related Guides

- [Choosing an Encoder](choosing_encoder.md)
- [Debugging Memory Issues](debugging_memory.md)
- [Choosing a Loss Function](choosing_loss_function.md)
