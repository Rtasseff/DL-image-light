# Decision Guide: Choosing an Encoder

**Context**: You're configuring `model.encoder` (and optionally `encoder_weights`) in your YAML config.

**Time to read**: 5 minutes

---

## Quick Decision Flow

```
Start with resnet34 (default)
   ↓
Is GPU memory < 8 GB or training slow?
   ├─ YES → Use "resnet18"
   └─ NO → Continue
       ↓
Need higher accuracy after fixing data & loss?
       ├─ YES → Try "resnet50"
       └─ NO → Stay with "resnet34"
```

---

## Encoder Comparison

| Encoder | Params | Speed | Memory Footprint | Best When |
|---------|--------|-------|------------------|-----------|
| **resnet18** | ~11M | Fastest | Low | Low-memory GPUs/CPUs, quick experiments |
| **resnet34** | ~21M | Fast | Moderate | Balanced choice for most workloads |
| resnet50 | ~23M | Medium | Higher | Need more capacity after tuning |
| efficientnet-b0 | ~4M | Fast | Low | Edge deployment, CPU inference |

---

## Recommendations

- Stick with **resnet34** until you have baseline metrics; it matches Golden Path defaults and pre-trained weights are reliable.
- Drop to **resnet18** if you see OOM errors even after lowering batch size, or if epochs take too long; expect a small accuracy hit (~0.01 Dice).
- Move to **resnet50** only after confirming data quality and loss choice; it increases compute time but captures more complex features.
- Use **efficientnet-b0** for edge or CPU-bound inference; training can be slower without GPU acceleration.
- Always set `encoder_weights: "imagenet"` unless you have domain-specific pre-training; this speeds up convergence.

---

## Config Snippets

```yaml
# Default baseline
model:
  architecture: "Unet"
  encoder: "resnet34"
  encoder_weights: "imagenet"
```

```yaml
# Low-memory setup
model:
  architecture: "Unet"
  encoder: "resnet18"

training:
  batch_size: 4  # adjust with memory savings
```

```yaml
# Accuracy-focused run
model:
  architecture: "UnetPlusPlus"
  encoder: "resnet50"

training:
  epochs: 80  # give the deeper encoder time to converge
```

---

## Metrics & Diagnostics

- If **val_dice** improves <0.01 after swapping encoders, revert to the simpler option.
- Monitor GPU RAM via `nvidia-smi`; resnet50 typically adds ~1 GB vs resnet34 at batch size 8.
- When encoder depth increases, watch for overfitting (gap between train/val metrics); consider data augmentation or regularization.

---

## When to Escalate

- You need encoders outside SMP’s catalog → raise an engineering task to extend the factory safely.
- Transfer learning is required from a custom checkpoint → coordinate with core developers; this may impact interface contracts.
- You want mixed-precision inference or quantization → wait for ADVANCED deployment docs or engage the architecture team.

---

## Related Guides

- [Choosing an Architecture](choosing_architecture.md)
- [Debugging Memory Issues](debugging_memory.md)
- [Choosing a Loss Function](choosing_loss_function.md)
