# For Engineers: How to Use This SDD

**If you're implementing features or fixing bugs, read this first.**

---

## Document Hierarchy (What to Read When)

```
USER asking "How do I segment images?"
   → README.md (2 min)
   → GOLDEN_PATH.md (15 min)
   
USER asking "Should I use focal or dice loss?"
   → DECISION_GUIDES/choosing_loss_function.md (5 min)
   
ENGINEER implementing new feature
   → SDD.md Sections 4 (Contracts), 6 (Fallback Policy), 9 (Testing)
   → Relevant CONTRACTS/*.md
   
ENGINEER fixing bug
   → SDD.md Section 4 (find which contract is violated)
   → Check test in tests/unit/ or tests/integration/
   
ENGINEER confused about requirement
   → SDD.md Section 1-2 (Purpose & Scope)
   → Ask: "Is this in scope or future work?"
   
ENGINEER onboarding
   → README.md → GOLDEN_PATH.md → SDD.md Sections 1-5
   → Skip sections 7-15 until you need them
```

---

## The SDD is NOT a Tutorial

**What the SDD is**:
- ✅ **Architecture specification** - What contracts must remain stable
- ✅ **Scope definition** - What we build vs. what we don't
- ✅ **Policy documentation** - Rules for fallbacks, validation, testing
- ✅ **Extension guide** - How to add new models, losses, metrics
- ✅ **Reference manual** - Look up specific requirements

**What the SDD is NOT**:
- ❌ User guide (that's GOLDEN_PATH.md)
- ❌ Quick start (that's README.md)
- ❌ Tutorial on segmentation (that's external)
- ❌ Decision guide (that's DECISION_GUIDES/)
- ❌ API documentation (that's in docstrings)

---

## Key Principle: Separation of Concerns

The SDD grew to 180 pages because we kept mixing:
1. **User instructions** ("how to train a model")
2. **Engineering contracts** ("this interface must not change")
3. **Decision guidance** ("which loss function to use")
4. **Implementation details** ("how auto-tuning works")

**Now they're separated**:

| Question | Document | Who |
|----------|----------|-----|
| How do I use this? | README.md, GOLDEN_PATH.md | Users |
| Which option should I choose? | DECISION_GUIDES/ | Users |
| What must remain stable? | CONTRACTS/ | Engineers |
| How does X work internally? | SDD.md Sections 7+ | Engineers |
| What's the full specification? | SDD.md (all) | Reviewers, Architects |

---

## Critical SDD Sections for Engineers

### Section 4: Architecture Contracts ⭐⭐⭐ CRITICAL
**These CANNOT change without major version bump.**

```python
# These interfaces in Section 4 are FROZEN
class DataModule(Protocol):      # Section 4.1
class Model(nn.Module):          # Section 4.2
class Loss(Protocol):            # Section 4.3
class Metrics(Protocol):         # Section 4.4
class Trainer(Protocol):         # Section 4.5
```

**Before changing any of these**:
1. Check if it breaks the contract
2. If yes → Requires SDD v5.0 (major version)
3. If no → Safe to implement

**Example**:
```python
# ❌ BREAKS CONTRACT (changes signature)
def train_dataloader(self, shuffle: bool = True) -> DataLoader:

# ✅ OK (internal implementation)
def _create_train_dataset(self, augment: bool) -> Dataset:
```

### Section 4.6: Construction Factories ⭐⭐⭐ CRITICAL
**All model/checkpoint construction goes through factories.**

**Rules**:
1. **Models**: Only `src/models/factory.build_model(config)` creates models
2. **Scripts**: Never import `segmentation_models_pytorch` directly
3. **Checkpoints**: Use `src/core/checkpoints.build_checkpoint_callback()`

**Tests check this**:
- `tests/unit/test_model_factory.py` - Factory works
- `tests/unit/test_scripts_imports.py` - Scripts don't violate rules

### Section 6: Fallback Policy ⭐⭐ IMPORTANT
**Fallbacks are for testing only, never production.**

When adding dependencies:
```python
# ✅ Correct pattern
try:
    from torchmetrics import Dice
    HAVE_TORCHMETRICS = True
except ImportError:
    if should_use_fallbacks():
        from ..testing.simple_metrics import SimpleDice as Dice
        HAVE_TORCHMETRICS = False
    else:
        raise ImportError("torchmetrics required. Install: pip install torchmetrics")
```

### Section 9: Testing Strategy ⭐⭐ IMPORTANT
**Three test tiers - know which one you're writing.**

```python
# Unit test - No ML dependencies
# tests/unit/test_config.py
def test_config_validation():
    config = {"model": {"architecture": "Unet"}}
    # Test pure logic

# Integration (minimal) - Core dependencies, fallbacks OK
# tests/integration/minimal/test_pipeline.py
@pytest.mark.minimal
def test_training_pipeline():
    # Uses simple implementations

# Integration (full) - All dependencies, no fallbacks
# tests/integration/full/test_training_full.py
@pytest.mark.full
def test_training_with_smp():
    # Uses real SMP models
```

---

## Common Engineering Questions

### "Can I add a new model architecture?"

**Yes**, via the factory:

1. Check if SMP supports it: https://smp.readthedocs.io/
2. If yes:
   ```python
   # src/models/factory.py
   def build_model(config):
       if architecture == "newarch":
           return smp.NewArch(...)
   ```
3. Add config schema in `src/core/config.py`
4. Add test in `tests/unit/test_model_factory.py`
5. Add to DECISION_GUIDES/choosing_architecture.md

**Don't**: Implement the architecture from scratch (violates "don't reinvent wheel")

### "Can I add a new loss function?"

**Yes**, this is an exception to "don't reinvent":

1. Loss functions are mathematical formulas (OK to implement)
2. Add to `src/losses/` following existing pattern
3. Register in `src/losses/__init__.py:get_loss()`
4. Add tests in `tests/unit/test_losses.py`
5. Document in DECISION_GUIDES/choosing_loss_function.md

**Why exception?** SMP doesn't provide losses, and these are well-defined math, not complex architectures.

### "Can I change the config schema?"

**Depends**:

```yaml
# ✅ Adding new optional field - OK
model:
  architecture: "Unet"
  new_optional_param: 123  # Default to None, backward compatible

# ⚠️ Renaming field - Needs migration guide
model:
  classes: 1  # OLD
  out_channels: 1  # NEW - provide migration script

# ❌ Removing required field - Breaks compatibility
model:
  encoder: "resnet34"  # Don't remove this
```

**Process**:
1. Add new field to schema in `src/core/config.py`
2. Update `configs/base_config.yaml` with example
3. Add validation test
4. Update GOLDEN_PATH.md with new option

### "The SDD says don't do X, but I need to do X"

**Process**:
1. Check if it's actually in scope (Section 2)
2. If out of scope → Discuss if scope should change
3. If in scope but forbidden → Understand why (read relevant section)
4. If justified → Propose SDD change (increment version)

**Example**:
- **In scope**: "Add new encoder from timm library" → Allowed, use factory
- **Out of scope**: "Add object detection" → This is segmentation only
- **Forbidden**: "Import SMP in script" → Violates factory pattern

### "Which validation mode should my test use?"

```python
# Unit tests
os.environ['CONFIG_VALIDATION_MODE'] = 'MINIMAL'

# Integration tests (minimal)
os.environ['CONFIG_VALIDATION_MODE'] = 'MINIMAL'

# Integration tests (full)
os.environ['CONFIG_VALIDATION_MODE'] = 'STRICT'

# Production (default)
# No environment variable needed, STRICT is default
```

---

## Red Flags (Stop and Ask)

🚩 **You're importing `segmentation_models_pytorch` in a script file**
   → Use factory instead

🚩 **You're implementing a new U-Net architecture**
   → Use SMP's Unet, don't reinvent

🚩 **You're creating `ModelCheckpoint` in multiple places**
   → Use `checkpoints.build_checkpoint_callback()`

🚩 **You're adding a fallback that will be used in production**
   → Fallbacks are test-only

🚩 **You're changing a method signature in Section 4 contracts**
   → This breaks compatibility, needs major version bump

🚩 **You're adding configuration that users must understand**
   → Document in GOLDEN_PATH.md and DECISION_GUIDES/

---

## Before You Submit a PR

### Checklist:
- [ ] Code follows factory patterns (Section 4.6)
- [ ] No contract interfaces changed (Section 4.1-4.5)
- [ ] Tests added for new functionality (Section 9)
- [ ] Config changes documented in GOLDEN_PATH.md
- [ ] User-facing decisions added to DECISION_GUIDES/
- [ ] SDD.md updated if policies changed
- [ ] No `import segmentation_models_pytorch` in scripts/
- [ ] `make test` passes

### When to Update SDD:
- ✅ New architectural pattern introduced
- ✅ New policy added (e.g., caching strategy)
- ✅ Contract extended (new optional method)
- ❌ Bug fix (just fix it)
- ❌ Performance optimization (just do it)
- ❌ Documentation improvement (update docs directly)

---

## For Code Reviewers

### Review Priorities:

1. **Contract Stability** (⭐⭐⭐)
   - Check Section 4 interfaces unchanged
   - Verify factory pattern usage

2. **Testing** (⭐⭐⭐)
   - New code has appropriate tier tests
   - Tests don't violate tier boundaries

3. **User Impact** (⭐⭐)
   - Config changes are backward compatible
   - GOLDEN_PATH.md updated if user-visible
   - Decision guides updated if options added

4. **SDD Compliance** (⭐⭐)
   - No SMP imports in scripts
   - Fallbacks used correctly
   - Validation modes respected

5. **Code Quality** (⭐)
   - Normal code review stuff
   - Performance, readability, etc.

---


---

## Operational Workflow (Docs + Agents)

1. **Start with the right context**: link agents or reviewers to `README_STREAMLINED.md` for user tasks, `GOLDEN_PATH.md` for end-to-end walkthroughs, and the specific decision guide that matches the issue.
2. **Log new blockers immediately**: capture reproducible steps in `docs/MANUAL_TESTS.md` before applying workarounds. Add a short note in the README "When Something Goes Wrong" table or create a new decision guide entry pointing back to the manual test.
3. **Plan new feature work**: open a note in `docs/ACTION_PLAN.md` with problem, goal, and impacted components. Reference any contracts (`CONTRACTS/`) or advanced topics (`ADVANCED/`) you expect to touch.
4. **Update policy docs before code changes**: extend `FOR_ENGINEERS.md`, `GOLDEN_PATH.md`, or the relevant decision guide with the rules your change introduces so agents have guidance before editing code.
5. **Ship with validation details**: append verification steps and expected outcomes to the same entry in `docs/MANUAL_TESTS.md` once the feature is merged; link to new documentation fragments and tests.
6. **Coordinate through PRs/issues**: call out documentation deltas in PR descriptions and point reviewers to the affected guides.

Following this loop keeps the navigation layer trustworthy and gives agents a single source for deltas before they execute changes.

---
## Summary: Your Mental Model

```
SDD.md = "The Constitution"
   ↓
Defines what MUST remain stable (contracts)
Defines what's allowed vs forbidden (policies)
Defines how to extend (patterns)
   ↓
NOT a tutorial, NOT a user guide
   ↓
Read when:
- Implementing features touching core contracts
- Unsure if something is allowed
- Need to understand system architecture
- Writing tests (which tier?)
   ↓
Don't read when:
- Just using the platform (read README/GOLDEN_PATH)
- Deciding which model to use (read DECISION_GUIDES)
- Writing application code (read API docs)
```

**The SDD is for builders, not users.**

---

## Questions?

- **"Is X in scope?"** → SDD Section 2
- **"Can I change Y?"** → SDD Section 4 (contracts)
- **"How should I implement Z?"** → SDD Section 7+ (implementation guides)
- **"Why does this rule exist?"** → Read the section, ask in discussions
- **"This seems over-engineered"** → It might be! Propose simplification with rationale

**Remember**: The SDD grew from pain points. Every rule has a "we got burned" story behind it.
