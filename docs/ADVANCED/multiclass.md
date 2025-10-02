# Advanced: Multi-Class Segmentation Status

**Current state**: The MVP ships single-class segmentation (background vs foreground). Multi-class support is not production-ready but you can prototype cautiously.

---

## Experimental Approach

1. **Adjust configuration**:
   ```yaml
   model:
     architecture: "Unet"
     encoder: "resnet34"
     out_channels: <number_of_classes>

   training:
     loss:
       type: "cross_entropy"
   ```
2. **Update dataset masks** to use integer labels `[0..N-1]` per pixel.
3. **Metrics**: Implement or enable multi-class metrics (e.g., per-class Dice) before trusting results.
4. **Evaluation**: Confirm `scripts/evaluate.py` handles multi-channel outputs; extend if necessary.

---

## Known Gaps

- Decision guides and Golden Path assume binary masks.
- Visualization overlays currently render a single class; multi-class palettes need implementation.
- Loss library includes dice/focal variants tuned for binary segmentation; multi-class equivalents require testing.

---

## Recommendations

- Keep experiments isolated (new branch + config path) and mark results as exploratory.
- Add unit/integration tests for any tooling changes before merging to main.
- Document contract impacts; increasing `out_channels` may ripple into metrics and evaluation scripts.

---

## Escalation Criteria

- Production requirement for multi-class segmentation → initiate formal scope discussion and SDD update.
- Need for class-specific metrics or post-processing → involve analytics + visualization teams.

---

## Related Docs

- [Configuration Schema](../CONTRACTS/configuration_schema.md)
- [Stable Interfaces](../CONTRACTS/stable_interfaces.md)
