# Manual Validation Checklist

Use this checklist to confirm the segmentation platform is ready before larger experiments. Commands assume you are inside the project root.

## 0. Prerequisites
- Python 3.10+ available
- (Optional but recommended) fresh virtual environment activated
- GPU drivers ready if you intend to use CUDA

## 1. Install Golden Path Dependencies
```bash
pip install -r requirements/ml.txt
```
Verify install completes without resolver errors. This pulls the exact versions required by SDD v4.1.

## 2. Dependency and Environment Diagnostics
```bash
python scripts/check_dependencies.py
```
Expected result: final line reports `System ready for SDD v4.1 Golden Path`. Review warnings; resolve any ❌ results before proceeding.

## 3. Configuration Sanity Check
```bash
python - <<'PY'
from src.core.config import load_and_validate_config
cfg = load_and_validate_config('configs/base_config.yaml')
print('Config OK, model:', cfg['model']['architecture'])
PY
```
Confirm it prints the architecture and exits without exceptions.

## 4. Smoke Training Run (Tiny Dataset)
Use the placeholder data in `data/images` and `data/masks` to ensure the training loop runs end-to-end.
```bash
python scripts/train.py --config configs/base_config.yaml --name smoke_test
```
Watch for:
- `Starting training` log in the console
- `runs/smoke_test_*/` directory containing `train.log`, `config.yaml`, and `checkpoints/best.ckpt`
- `effective_settings.json` listing the checkpoint settings and platform info

If you need a quicker run, add `--fast-dev-run` (works once SDD wrapper adds support) or temporarily reduce `training.epochs` to 1 in a copy of the config.

## 5. Data-Driven Test (DRIVE Dataset)
Ensure the DRIVE dataset is available under `data/drive/` with the same structure configured in `configs/drive.yaml`. Then run:
```bash
python scripts/train.py --config configs/drive.yaml --name drive_manual
```
Confirm:
- Dataset counts print the expected 20 training / 5 validation images (default split)
- `runs/drive_manual_*/` contains checkpoints and `dataset_info.json`
- Lightning logs show metric updates (`val/dice`, `val/loss`)

If the dataset lives elsewhere, update `configs/drive.yaml` paths or provide `--config` pointing to a copy with correct directories.

### Obtaining the DRIVE Dataset
1. Download the official ZIP from the [DRIVE challenge site](https://drive.grand-challenge.org/).
2. Extract so that you have `training/images`, `training/1st_manual`, and `test/images` folders.
3. Copy or symlink the root into `data/drive/` (i.e., the training images live at `data/drive/training/images`).
4. Re-run step 5 to confirm the data module reports the correct sample counts.

### Optional Metric Spot Check
After the run finishes, open `runs/drive_manual_*/train.log` and confirm `val/dice` reaches at least ~0.70 by epoch 50 (baseline expectation from the SDD MVP). If results are significantly lower, double-check augmentations and data paths.

## 6. Checkpoint Single-Source Verification
Inspect the trainer to ensure only one `ModelCheckpoint` is active:
```bash
python - <<'PY'
from pathlib import Path
from src.core.config import load_and_validate_config
from src.core.trainer import SegmentationTrainer
from src.models.factory import build_model
from src.models.lightning_module import SegmentationModel
from src.data.datamodule import SegmentationDataModule

cfg = load_and_validate_config('configs/base_config.yaml')
run_dir = Path('runs/manual_checkpoint_check')
run_dir.mkdir(parents=True, exist_ok=True)
trainer = SegmentationTrainer(cfg, run_dir, 'checkpoint_check')
model = SegmentationModel(cfg, build_model(cfg))
dm = SegmentationDataModule(cfg)
dm.setup('fit')
trainer_obj = trainer._create_trainer()
count = sum(1 for cb in trainer_obj.callbacks if cb.__class__.__name__ == 'ModelCheckpoint')
print('ModelCheckpoint callbacks:', count)
PY
```
Expected output: `ModelCheckpoint callbacks: 1`.

## 7. Inference Sanity Check
After a training run, load the best checkpoint and run validation/prediction to verify the forward path:
```bash
python - <<'PY'
import torch
from pathlib import Path
from pytorch_lightning import Trainer
from src.core.config import load_and_validate_config
from src.models.factory import build_model
from src.models.lightning_module import SegmentationModel
from src.data.datamodule import SegmentationDataModule

run_dir = sorted(Path('runs').glob('drive_manual_*'))[-1]
cfg = load_and_validate_config(run_dir / 'config.yaml')
model = SegmentationModel(cfg, build_model(cfg))
ckpt = run_dir / 'checkpoints' / 'best.ckpt'
trainer = Trainer(logger=False, enable_checkpointing=False, devices=1, accelerator='cpu')
dm = SegmentationDataModule(cfg)
dm.setup('fit')
metrics = trainer.validate(model, datamodule=dm, ckpt_path=str(ckpt))
print(metrics)
PY
```
Verify metrics print as a list of dictionaries and no errors are thrown.

---
Running each step in order gives high confidence that the platform respects the SDD v4.1 contracts and functions on real data. Record any deviations or warnings before moving on to larger experiments.
