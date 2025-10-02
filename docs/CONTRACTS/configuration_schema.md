# Configuration Schema Overview

**Purpose**: Summarize required configuration fields, defaults, and safe extension points.

---

## Required Top-Level Sections

| Section | Required Keys | Notes |
|---------|---------------|-------|
| `project_name` | string | Used for run directories and logging. |
| `dataset` | `images_dir`, `masks_dir` | Paths must exist before training. |
| `model` | `architecture`, `encoder`, `out_channels` | Architecture/encoder names must be supported by the factory. |
| `training` | `epochs`, `batch_size`, `learning_rate`, `loss.type` | Loss type must match registered losses. |
| `output` | `dir`, `checkpoint.monitor`, `checkpoint.mode` | Controls run artifacts and best-checkpoint policy. |
| `compute` | `accelerator`, `precision`, `seed` | Defaults align with the Golden Path. |

Optional sections (`resources`, `visualization`, `dataset.augmentation`, etc.) can be omitted; validation fills defaults depending on mode.

---

## Golden Path Baseline

```yaml
model:
  architecture: "Unet"
  encoder: "resnet34"
  out_channels: 1
  encoder_weights: "imagenet"

training:
  epochs: 50
  batch_size: 8
  learning_rate: 1e-4
  loss:
    type: "dice"
    params:
      smooth: 1.0
  metrics: ["dice", "iou"]
```

---

## Extending the Schema Safely

1. **Add optional fields only**: provide sensible defaults in `src/core/config.py`.
2. **Update `configs/base_config.yaml`** with an example value.
3. **Document user-facing changes** in `docs/GOLDEN_PATH.md` and relevant decision guides.
4. **Add validation tests** (unit + integration) to cover new paths.
5. **Avoid renaming fields**; if unavoidable, supply migration notes and maintain backward compatibility where possible.

---

## Common Pitfalls

- Forgetting to set `model.out_channels` when experimenting—validation will fail in STRICT mode.
- Adding required keys without updating docs; this breaks analysts following README instructions.
- Diverging configs between scripts and library code; always reference `runs/<experiment>/config.yaml` for provenance.

---

## Related References

- Golden Path configuration: `docs/GOLDEN_PATH.md`
- Base example: `configs/base_config.yaml`
- Validation logic: `src/core/config.py`
