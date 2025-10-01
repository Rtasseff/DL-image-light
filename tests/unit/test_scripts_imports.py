"""Guards around script import policies."""

from pathlib import Path


def test_train_script_does_not_import_smp():
    train_file = Path("scripts/train.py")
    content = train_file.read_text()
    assert "segmentation_models_pytorch" not in content
