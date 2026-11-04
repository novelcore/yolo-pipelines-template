# model-registration

Registers the trained `best.pt` (and `last.pt`) in the MLflow model registry.

- **Runs on:** CPU, always, last.
- **Reads:** `dataset`, `data`, `model`, `registration` from the form, and `training-result` (and the quantization result when `quantization-mode` is not `none`).
- **Does:** registers a new version of `registration-registered_model_name` (default `object-pose-yolo`) from the training run's logged model, tags the lakeFS commit `model-<name>-v<N>`, sets the `kaos.*` tags on the version, and gives it the alias in `registration-promote_to` (`champion` or `challenger`) if set.
- **Output:** `registration-result`.
- **Settings:** [docs/parameters.md](../../docs/parameters.md) (model: `app/models/registration.py`).

```
steps/model_registration/
  app/entry.py                     reads params and upstream results
  app/manager.py                   orchestrates registration
  app/models/registration.py       RegistrationParams, RegistrationResult
  app/services/model_registration.py   MLflow and lakeFS calls
```

Pick and promote models: [docs/promote.md](../../docs/promote.md).
