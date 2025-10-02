# Validation & Fallback Rules

**Purpose**: Define configuration validation modes and when fallbacks are permitted.

---

## Validation Modes

| Environment | Mode | Behavior |
|-------------|------|----------|
| Local development | PERMISSIVE | Missing optional fields get defaults + warnings |
| CI (unit tests) | MINIMAL | Only essential fields required |
| CI (integration) | STRICT | Full validation, no missing fields |
| Production / Releases | STRICT | Full validation, fail fast on issues |

### Mode Selection

```python
mode = get_validation_mode()
# Priority order:
# 1. CONFIG_VALIDATION_MODE env override
# 2. CI detection (STRICT unless unit suite)
# 3. Local development (PERMISSIVE)
# 4. Default to STRICT
```

**Operator actions**:
- Override with `CONFIG_VALIDATION_MODE=strict` when forcing strict checks locally.
- Keep production environments at STRICT; this is non-negotiable per SDD v4.1.

---

## Fallback Policy

| Context | Fallbacks Allowed? | How |
|---------|--------------------|-----|
| Unit tests | ✅ Yes | Automatic via `should_use_fallbacks()` |
| Integration (minimal) | ✅ Yes | Automatic via test markers |
| Local development | ⚠️ Optional | `export USE_FALLBACKS=true` (temporary) |
| CI full, production, releases | ❌ No | Never enable fallbacks |

**Implementation pattern**:

```python
try:
    from torchmetrics import Dice
except ImportError:
    if should_use_fallbacks():
        from ..testing.simple_metrics import SimpleDice as Dice
    else:
        raise ImportError("torchmetrics required. Install via requirements/ml.txt")
```

- Fallbacks exist solely for test environments where dependencies are intentionally pruned.
- Any production run with fallbacks enabled must be treated as a defect.

---

## Compliance Checklist

- [ ] No scripts set `USE_FALLBACKS=true` by default.
- [ ] Production configs omit `use_fallbacks` or set to `false`.
- [ ] Tests document which validation mode they expect (unit → MINIMAL, full → STRICT).
- [ ] `CONFIG_VALIDATION_MODE` overrides are used sparingly and never committed to `.env` files.

---

## Related References

- Validation code: `src/core/config.py`
- Fallback helper: `src/core/dependencies.py`
- Policy summary: `docs/FOR_ENGINEERS.md`
