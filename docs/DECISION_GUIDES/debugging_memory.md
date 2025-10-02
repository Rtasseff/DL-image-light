# Decision Guide: Debugging GPU/Memory Issues

**Context**: Training fails with out-of-memory (OOM) errors or exhausts system memory.

**Time to read**: 5 minutes

---

## Immediate Checklist (Try in Order)

1. **Lower `training.batch_size`**
   ```yaml
   training:
     batch_size: 4  # try 4 → 2 → 1
   ```
2. **Switch to a lighter encoder** (see `choosing_encoder.md`)
   ```yaml
   model:
     encoder: "resnet18"
   ```
3. **Disable heavy augmentations** temporarily
   ```yaml
   dataset:
     augmentation:
       enabled: false
   ```
4. **Reduce output artifacts**
   ```yaml
   output:
     save_overlays: false
   ```
5. **Close other GPU processes** before rerunning (`nvidia-smi` or Activity Monitor).

---

## Diagnostic Questions

- Does the error appear at the **start of training**? → Model + batch size too large.
- Does it appear **mid-epoch**? → Augmentations or caching may spike memory; disable transforms.
- Are you running on **CPU only**? → Lower `training.num_workers` and confirm images are resized appropriately.

---

## Advanced Tweaks (If Above Fails)

- Switch to **mixed precision** when hardware supports it:
  ```yaml
  compute:
    precision: 16
  ```
  *Warning*: Confirm Lightning and drivers support amp; revert if NaNs appear.
- Crop or resize images offline; aim for ≤512×512 during experimentation.
- Split data into smaller subsets to validate that configuration, not data volume, is the issue.

---

## When to Escalate

- OOM persists at `batch_size: 1` **and** with `resnet18` → flag to engineering; may require gradient accumulation or architectural changes.
- Memory spikes when logging overlays even after disabling them → open an issue with config + sample images.
- Mixed precision causes instability that you cannot resolve → reset to `precision: 32` and escalate.

---

## Related Guides

- [Choosing an Encoder](choosing_encoder.md)
- [Debugging Performance](debugging_performance.md)
- [Golden Path Walkthrough](../GOLDEN_PATH.md)
