# Stable Interfaces (v4.1 – v5.0)

**Purpose**: Document the five contracts that may not change without a major version bump.

---

## What Is Locked

- Interfaces defined in SDD Section 4 stay stable through **v5.0**.
- Breaking any method signature or required behavior requires an SDD major version update and architecture review.
- Implementations may change internally, but factories and trainers must continue to honor these signatures.

---

## Contracts at a Glance

```python
class DataModule(Protocol):
    def setup(self, stage: str) -> None: ...
    def train_dataloader(self) -> DataLoader: ...
    def val_dataloader(self) -> DataLoader: ...
    def test_dataloader(self) -> Optional[DataLoader]: ...
```
- **Guarantees**: Dataloaders exist; batching, transforms, and shuffling are internal details.
- **Allowed changes**: New helper methods, alternate dataset objects.
- **Forbidden**: Renaming loaders, altering required return types.

```python
class Model(nn.Module):
    def forward(self, x: Tensor) -> Tensor: ...
    def predict_step(self, x: Tensor, strategy: str = "standard") -> Tensor: ...
```
- **Guarantees**: Forward outputs logits/probabilities; `predict_step` supports strategy flag even if only `"standard"` implemented today.
- **Allowed**: Internal modules, hooks, logging.
- **Forbidden**: Changing signature or removing `strategy` argument.

```python
class Loss(Protocol):
    def __call__(self, pred: Tensor, target: Tensor, **kwargs) -> Tensor: ...
```
- **Guarantees**: Callable returns scalar tensor; supports keyword overrides.
- **Allowed**: New optional kwargs, composition wrappers.
- **Forbidden**: Returning non-tensor values, mutating inputs.

```python
class Metrics(Protocol):
    def update(self, pred: Tensor, target: Tensor) -> None: ...
    def compute(self) -> Dict[str, Tensor]: ...
    def reset(self) -> None: ...
```
- **Guarantees**: Stateless compute; reset clears accumulators.
- **Allowed**: Additional helper methods.
- **Forbidden**: Changing compute return type or reset signature.

```python
class Trainer(Protocol):
    def fit(self, model, datamodule, callbacks=None) -> None: ...
    def validate(self, model, datamodule) -> List[Dict]: ...
    def test(self, model, datamodule) -> List[Dict]: ...
```
- **Guarantees**: Return lists of dictionaries; accepts callbacks list.
- **Allowed**: Additional optional kwargs if defaulted.
- **Forbidden**: Requiring new positional arguments, changing return types.

---

## Factory Rules

- All models originate from `src/models/factory.build_model(config)`.
- Scripts must **not** import `segmentation_models_pytorch` directly; factories provide the abstraction layer.
- Only one `ModelCheckpoint` is created via `src/core/checkpoints.build_checkpoint_callback()`.

---

## Compliance Checklist

- [ ] No direct SMP imports in scripts.
- [ ] Config-driven changes only; code branching by architecture/encoder is handled inside factories.
- [ ] Tests `tests/unit/test_model_factory.py` and `tests/unit/test_scripts_imports.py` remain green.
- [ ] Contract signatures untouched; any change triggers proposal for SDD v5.0.
