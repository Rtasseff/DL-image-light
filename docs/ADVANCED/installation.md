# Advanced Installation & Environment Notes

**Use this when** the standard `pip install -r requirements/ml.txt` path fails or you need a custom environment.

---

## Supported Environments

| Target | Status | Notes |
|--------|--------|-------|
| CUDA GPU (Linux/Windows) | ✅ Supported | Requires NVIDIA driver ≥ 525 and CUDA 12 toolkit. |
| Apple Silicon (MPS) | ✅ Supported | macOS 13+, Xcode Command Line Tools installed. |
| CPU-only | ⚠️ Supported for inference / tests | Expect slower training; use fallbacks cautiously. |

---

## Troubleshooting Steps

1. **Verify Python**: 3.10 recommended. Use `pyenv` to avoid system Python conflicts.
2. **Create virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   ```
3. **Install core stack**:
   ```bash
   pip install -r requirements/ml.txt
   ```
4. **If torch install fails**:
   - CUDA: match PyTorch wheels from https://pytorch.org/get-started/locally/
   - Apple Silicon: use `pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/nightly/cpu`
5. **Validate install**:
   ```bash
   python - <<'PY'
   import torch
   import pytorch_lightning as pl
   print(torch.cuda.is_available(), pl.__version__)
   PY
   ```

---

## Optional Extras

- **Dev tools**: `pip install -r requirements/dev.txt`
- **Loose pins** (experimentation): `pip install -r requirements/ml-loose.txt`
- **System packages** (Ubuntu): `sudo apt-get install libjpeg-dev zlib1g-dev`

---

## Known Issues

- Apple Silicon + PyTorch Lightning 2.1 occasionally logs warnings about unsupported ops; training still succeeds.
- Windows + CUDA may need the Visual Studio Build Tools.
- If you must use conda, install PyTorch via `conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia` then `pip install` the remaining packages.

---

## When to Ask for Help

- You need to compile PyTorch from source or support CUDA < 12 → open a platform ticket.
- You require multi-GPU setup → planned for v2.0, not MVP.
- Dependency conflicts remain after creating a fresh venv → share `pip list` and error logs with the team.
