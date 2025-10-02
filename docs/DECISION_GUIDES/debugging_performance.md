# Decision Guide: Debugging Model Performance

**Context**: Validation metrics (Dice/IoU) are lower than expected or regress over time.

**Time to read**: 5 minutes

---

## Start with Data Checks

1. **Verify masks match images**
   - Same filename stem and dimensions
   - Binary (0/255) for single-class tasks
2. **Inspect 10 random overlays**
   - Use `runs/<experiment>/visualizations/` or create manual overlays
3. **Check class balance**
   ```python
   from pathlib import Path
   import numpy as np
   from PIL import Image

   total_pixels = 0
   foreground_pixels = 0

   for mask_path in Path('data/masks').glob('*.png'):
       mask = np.array(Image.open(mask_path))
       total_pixels += mask.size
       foreground_pixels += (mask > 127).sum()

   ratio = foreground_pixels / total_pixels
   print(f"Foreground: {ratio*100:.2f}%")
   ```
   - If foreground <10%, consider focal or focal-tversky losses (see `choosing_loss_function.md`).

---

## Training Diagnostics

```
Look at metrics per epoch
   ↓
Is val_loss decreasing but metrics low?
   ├─ YES → Possible thresholding issue → inspect probability maps
   └─ NO → Continue
       ↓
Is training much better than validation?
       ├─ YES → Overfitting → add augmentation, reduce epochs
       └─ NO → Underfitting → adjust architecture/encoder
```

### Quick Fixes

- Increase training epochs to 75–100 if curves still improving.
- Lower learning rate (e.g., `5e-5`) if loss oscillates or diverges.
- Enable basic augmentation:
  ```yaml
  dataset:
    augmentation:
      enabled: true
      horizontal_flip_prob: 0.5
  ```
- Switch loss:
  - Thin structures → `focal`
  - Penalize false positives → `tversky` with `alpha: 0.7`

---

## Architecture & Encoder Tweaks

- Move from **Unet → UnetPlusPlus** when boundaries are rough or small features vanish.
- Try **resnet50** encoder if the model underfits after data fixes.
- Record every change as a new experiment; compare `overall_metrics.json` between runs.

---

## Evaluation Sanity Checks

- Confirm evaluation uses the same config saved during training (`runs/<experiment>/config.yaml`).
- Review per-image metrics (`evaluation/per_image_metrics.csv`) to see if failure is localized.
- Adjust prediction threshold for probability maps (e.g., rerun inference with `--threshold 0.4` and inspect overlays).

---

## When to Escalate

- Metrics stay <0.5 after data verification, multiple losses, and architecture swaps → request data science review.
- You need multi-class segmentation or custom metrics → coordinate with platform engineers; outside MVP scope.
- Model improves on train but not validation after numerous tweaks → capture config + sample batch and open an issue for deeper investigation.

---

## Related Guides

- [Choosing a Loss Function](choosing_loss_function.md)
- [Choosing an Architecture](choosing_architecture.md)
- [Golden Path Walkthrough](../GOLDEN_PATH.md)
