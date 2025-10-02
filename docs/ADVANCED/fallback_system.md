# Advanced: Fallback System Reference

**Purpose**: Explain how fallback implementations work and how to use them responsibly.

---

## What Is a Fallback?

Fallbacks are lightweight stand-ins for heavy dependencies (e.g., TorchMetrics implementations) used to keep unit/minimal integration tests fast and environment-independent.

---

## Decision Matrix

| Scenario | Use Fallbacks? | Action |
|----------|----------------|--------|
| Unit tests | ✅ Yes | Auto-enabled; no action required. |
| Minimal integration tests | ✅ Yes | Auto-enabled when running with `pytest -m minimal`. |
| Local debugging without full deps | ⚠️ Optional | `export USE_FALLBACKS=true` temporarily. |
| Production, full integration tests, releases | ❌ No | Remove env var; install full dependencies. |

---

## Import Pattern

```python
try:
    from torchmetrics import Dice
except ImportError:
    from src.testing.simple_metrics import SimpleDice as Dice
```

Wrap this pattern via helpers:

```python
from src.core.dependencies import should_use_fallbacks

if should_use_fallbacks():
    from src.testing.simple_metrics import SimpleDice as Dice
else:
    from torchmetrics import Dice
```

---

## Logging Expectations

- When fallbacks load, a warning is emitted (“Using fallback implementations…”).
- Training scripts should log the effective metrics implementation via `EffectiveSettingsLogger`.
- Any PR introducing new fallbacks must document them in SDD Section 6 and add tests under `tests/testing/`.

---

## Risks & Mitigations

- **Drift**: Fallback behavior may diverge from real dependency. Mitigate by unit testing fallback vs real metric where possible.
- **Accidental Production Use**: STRICT validation + CI checks block fallbacks in release pipelines; never bypass these safeguards.

---

## Related Docs

- [Validation Rules](../CONTRACTS/validation_rules.md)
- `src/core/dependencies.py`
- `tests/unit/test_dependencies.py`
