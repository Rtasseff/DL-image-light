# Advanced: Deployment & Serving Guidance

**Scope**: The MVP focuses on training and evaluation. This guide describes safe hand-off patterns for using trained checkpoints downstream.

---

## Recommended Workflow

1. **Train via Golden Path**.
2. **Evaluate** with `scripts/evaluate.py` to capture metrics and overlays.
3. **Package artifacts**:
   - `runs/<experiment>/checkpoints/best.ckpt`
   - `runs/<experiment>/config.yaml`
   - `evaluation_<experiment>/overall_metrics.json`
4. **Ship to consumers** as a bundle (zip/TAR) or register in your model registry.

---

## Batch Inference Script

Use `scripts/predict.py` for offline scoring.

```bash
python scripts/predict.py   --checkpoint runs/exp_001/checkpoints/best.ckpt   --config runs/exp_001/config.yaml   --input /path/to/new_images   --output predictions/exp_001
```

Outputs include binary masks, probability maps, and overlays. Share overlays for quick QA feedback.

---

## Serving Considerations

- **Real-time APIs** are out of scope for MVP; wrap `predict.py` in your own service if required.
- **Model size**: Convert to TorchScript with Lightning if deployment target needs it:
  ```bash
  python scripts/export_torchscript.py --checkpoint ...
  ```
  (Provide script path once implemented.)
- **Versioning**: Tie deployments to git commit + config hash from `experiment_identity.json`.
- **Monitoring**: Log Dice/IoU on hold-out sets before release; attach plots to release notes.

---

## Safety Checklist Before Release

- [ ] Evaluation metrics meet acceptance criteria.
- [ ] Training/eval done in STRICT validation mode with fallbacks off.
- [ ] Config + checkpoint stored in artifact repository.
- [ ] Rollback plan defined (previous checkpoint available).

---

## When to Escalate

- Need real-time throughput or scale-out inference → coordinate with platform infra team.
- Deployment target requires ONNX / TensorRT exports → open architecture request; conversions not validated yet.
- Consumers demand multi-class outputs → see `multiclass.md` and engage data science for scope change.
