# Advanced: Cross-Validation Workflow

**Goal**: Run N-fold cross-validation while keeping the Golden Path intact.

---

## Recommended Pattern

1. **Prepare folds** externally (e.g., with scikit-learn) and store fold manifests:
   ```python
   from sklearn.model_selection import KFold
   import json
   from pathlib import Path

   image_paths = sorted(Path('data/images').glob('*.png'))
   kf = KFold(n_splits=5, shuffle=True, random_state=42)

   for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
       payload = {
           'train': [str(image_paths[i]) for i in train_idx],
           'val': [str(image_paths[i]) for i in val_idx],
       }
       Path(f'folds/fold_{fold}.json').write_text(json.dumps(payload, indent=2))
   ```
2. **Create per-fold configs** referencing the manifest:
   ```yaml
   dataset:
     split:
       type: "manifest"
       manifest_path: "folds/fold_0.json"
   ```
3. **Automate training**:
   ```bash
   for fold in 0 1 2 3 4; do
     python scripts/train.py        --config configs/cv_base.yaml        --name cv_fold_${fold}        --extra dataset.split.manifest_path=folds/fold_${fold}.json
   done
   ```

---

## Aggregating Results

- Collect Dice/IoU from `runs/cv_fold_*/overall_metrics.json`.
- Use a notebook or script to compute mean/variance; store summary under `reports/cv_summary.json`.
- Inspect worst-performing fold overlays to uncover systematic issues.

---

## Guardrails

- Keep validation mode STRICT for every fold to ensure consistency.
- Disable auto-tuning during CV; variation hides true fold differences.
- Ensure random seeds differ across folds only through data split, not training configuration.

---

## When to Escalate

- Need stratified or grouped splits → extend manifest format and update config validation accordingly.
- Require automated fold orchestration on cluster → coordinate with ML Ops; outside MVP.

---

## Related Docs

- [Golden Path](../GOLDEN_PATH.md)
- [Configuration Schema](../CONTRACTS/configuration_schema.md)
