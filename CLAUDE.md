# Claude Code Assistant Instructions

## **CRITICAL: Always Check SDD Document First**

**BEFORE starting any work**, ALWAYS read the current SDD document:
- **sdd.md** - Contains the complete SDD v4.1 specification
- This document defines ALL requirements, architecture, and policies
- Any work must comply with SDD v4.1 standards
- If there's any conflict between instructions, the SDD takes precedence

## Virtual Environment Usage

**CRITICAL**: This project uses a virtual environment (`venv/`) with all required dependencies installed. You MUST always activate it before running any Python code or tests.

### Required Pattern for All Python Commands

```bash
# ALWAYS start Python commands with this activation
source venv/bin/activate && python your_command.py
```

### Installed Dependencies

The `venv/` contains all required packages:
- PyTorch Lightning 2.5.5
- PyTorch 2.8.0
- TorchMetrics 1.8.2
- Segmentation Models PyTorch 0.3.3
- Albumentations 1.4.21
- All other project dependencies

### Common Mistake to Avoid

❌ **WRONG**: Running Python without activating venv
```bash
python -c "import pytorch_lightning"  # Will fail with ModuleNotFoundError
```

✅ **CORRECT**: Always activate venv first
```bash
source venv/bin/activate && python -c "import pytorch_lightning"  # Works correctly
```

### Testing and Development

When running tests, compliance checks, or any Python code:

```bash
# For single commands
source venv/bin/activate && python test_script.py

# For interactive sessions
source venv/bin/activate
python
>>> import pytorch_lightning  # Now works
```

### Project Structure

```
/Users/rtasseff/projects/DL-image-light/
├── venv/                    # Virtual environment with all dependencies
├── src/                     # Source code
├── configs/                 # Configuration files
├── requirements/            # Dependency specifications
└── CLAUDE.md               # This file
```

## SDD v4.1 Compliance

This project follows SDD v4.1 standards - **READ sdd.md FOR COMPLETE DETAILS**:

### Key Principles
- **Golden Path**: SMP + Lightning + Full dependencies + STRICT validation
- **Gated Complexity**: Fallbacks and auto-tuning are opt-in only
- **Explicit Over Magic**: All automatic behaviors must be logged
- **Version Pinning**: Exact versions for reproducibility

### Five Stable Interfaces (v4.0-v5.0)
1. **DataModule**: setup(), train_dataloader(), val_dataloader(), test_dataloader()
2. **Model**: forward(), predict(x, strategy="standard") with TTA support
3. **Loss**: __call__(pred, target) factory pattern
4. **Metrics**: update(), compute() -> Dict[str, Tensor], reset()
5. **Trainer**: fit(), validate(), test() Lightning wrapper

### Configuration Policy
- **STRICT mode**: Default validation (production)
- **PERMISSIVE mode**: Development with warnings
- **MINIMAL mode**: Unit tests only
- **Never use fallbacks in production**: use_fallbacks: false

### Validation Mode Detection
```bash
# Explicit override
export CONFIG_VALIDATION_MODE=STRICT|PERMISSIVE|MINIMAL

# Auto-detection:
# - CI: STRICT (unless unit tests = MINIMAL)
# - Local: PERMISSIVE
# - Default: STRICT
```

## Development Commands

```bash
# Run SDD compliance tests
source venv/bin/activate && python -c "from src.core import load_config; print('SDD v4.1 compliant')"

# Run training
source venv/bin/activate && python scripts/train.py --config configs/base_config.yaml

# Run validation
source venv/bin/activate && python scripts/validate.py --config configs/base_config.yaml
```

## Troubleshooting

### Virtual Environment Issues
Always check that you're using the virtual environment:
```bash
source venv/bin/activate && python -c "import sys; print(f'Python path: {sys.executable}')"
# Should show: /Users/rtasseff/projects/DL-image-light/venv/bin/python
```

### Dependency Issues
```bash
# Check if you can follow the Golden Path
source venv/bin/activate && python scripts/check_dependencies.py

# If Golden Path fails, enable fallbacks for development only
export USE_FALLBACKS=true
export CONFIG_VALIDATION_MODE=PERMISSIVE
```

### SDD Compliance Testing
```bash
# Run the complete SDD v4.1 compliance test
source venv/bin/activate && python -c "
from src.core import load_config, should_use_fallbacks, get_validation_mode
from src.metrics import create_sdd_metrics
from src.models.lightning_module import SegmentationModel
print('✅ SDD v4.1 compliant!')
"
```

### Common Error Patterns
- **ModuleNotFoundError**: Forgot to activate venv (use `source venv/bin/activate &&`)
- **Config validation errors**: Check validation mode with `echo $CONFIG_VALIDATION_MODE`
- **Fallback warnings**: Check `echo $USE_FALLBACKS` (should be unset in production)

## Remember
1. **ALWAYS** read sdd.md before starting work
2. **ALWAYS** activate venv before Python commands
3. **NEVER** use fallbacks in production
4. **ALWAYS** use STRICT validation for production
5. **FOLLOW** the Golden Path unless you have a specific reason to deviate