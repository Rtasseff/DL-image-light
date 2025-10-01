"""Factories for common Lightning callbacks."""

from pathlib import Path
from typing import Any, Dict

from pytorch_lightning.callbacks import ModelCheckpoint


def build_checkpoint_callback(config: Dict[str, Any], run_dir: Path) -> ModelCheckpoint:
    """Return a configured ModelCheckpoint callback."""
    checkpoint_cfg = config["output"]["checkpoint"]
    checkpoint_dir = Path(run_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_best_only = checkpoint_cfg.get("save_best_only", True)
    save_top_k = 1 if save_best_only else checkpoint_cfg.get("save_top_k", -1)

    filename = checkpoint_cfg.get("filename", "best")

    return ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename=filename,
        monitor=checkpoint_cfg["monitor"],
        mode=checkpoint_cfg["mode"],
        save_top_k=save_top_k,
        save_last=True,
        auto_insert_metric_name=False,
    )
