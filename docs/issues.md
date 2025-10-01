possible issues

# check point redundancy 

What’s happening

If both scripts/train.py and src/core/* are importing ModelCheckpoint and creating their own instances, you risk:

Duplicate checkpoint callbacks attached to the same Trainer → extra I/O, confusing files, or even conflicting monitor/dirpath.

Drift: one place changes the filename/metric/path and the other doesn’t.

Hidden behavior: a future contributor “fixes” one copy and forgets the other.

Even if they’re currently identical, it’s redundant and fragile.

What “good” looks like

Single source of truth for checkpoints (and other callbacks) lives in src/ as a small factory.

Scripts are thin: they call the factory and pass its result into the Trainer.

Example pattern

In src/core/checkpoints.py:

from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

def build_checkpoint_callback(run_dir: Path, monitor: str, mode: str = "max", top_k: int = 3):
    ckpt_dir = Path(run_dir) / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"epoch={{epoch:04d}}-step={{step:06d}}-{monitor.replace('/','_')}={{{{{{monitor}}}}}:.3f}",
        monitor=monitor, mode=mode, save_top_k=top_k, save_last=True, auto_insert_metric_name=False
    )


In scripts/train.py:

from src.core.checkpoints import build_checkpoint_callback
ckpt_cb = build_checkpoint_callback(run_dir, monitor="val/dice", mode="max", top_k=3)
trainer = Trainer(callbacks=[ckpt_cb, *other_callbacks], ...)


Now there’s one place to change naming/paths/monitor metric.

How to fix your repo (quick migration plan)

Audit where ModelCheckpoint is created:

grep -n "ModelCheckpoint" scripts/train.py src/core/*.py


Pick the canonical spot (recommend: src/core/checkpoints.py as above).

Refactor:

Move any callback creation logic from scripts/train.py into the factory.

If src/core/trainer.py was also creating a checkpoint callback internally, remove it and have it accept callbacks as an argument, or let train.py build the list and pass them in.

Unit test guard:

Add a test that builds the Trainer and asserts exactly one ModelCheckpoint is present:

from pytorch_lightning.callbacks import ModelCheckpoint
assert sum(isinstance(c, ModelCheckpoint) for c in trainer.callbacks) == 1


Effective settings log:

Record checkpoint.monitor, checkpoint.dirpath, save_top_k, save_last into effective_settings.json so drift is visible.

Deprecation note:

If you must keep a legacy creation path temporarily, add a runtime warning pointing to the new factory and an issue link; remove in a follow-up PR.

While you’re at it (same principle)

Do the same single-factory approach for other policy-heavy bits that mustn’t drift:

EarlyStopping callback

Learning rate schedulers

Loggers (CSV/W&B/MLflow) and their directories

Precision/devices/strategy defaults (centralize in a build_trainer(cfg) helper)

Quick acceptance checklist

Only one ModelCheckpoint instance reaches the Trainer.

The checkpoint dir is always runs/<run_id>/ckpts/.

best.ckpt and last.ckpt are always present.

The monitored metric name is identical across training/eval scripts.

A unit test enforces the “one checkpoint callback” rule.
