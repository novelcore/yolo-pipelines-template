# model-registration

Registers the trained `best.pt` (and `last.pt`) in the MLflow model registry.

- **Runs on:** CPU, always, last.
- **Reads:** the settings `data` and `registration`, and the results of `prepare-data`, `model-training` and, when it ran, `model-quantization` (`step.upstream(...)`): where each saved its files, with the exact commit.
- **Does:** shows `best.pt` in the experiment's Models tab as a logged model, registers a new version of `registration-registered_model_name` (default `object-pose-yolo`) from it, tags the lakeFS commit `model-<name>-v<N>`, sets the `kaos.*` tags on the version, and gives it the alias in `registration-promote_to` (`champion` or `challenger`) if set.
- **Hands on:** the registered versions.
- **Settings:** [docs/parameters.md](../../docs/parameters.md) (model: `app/models/registration.py`).

```
steps/model_registration/
  app/entry.py                     reads the results of the steps before it
  app/manager.py                   orchestrates registration
  app/models/registration.py       RegistrationParams, RegistrationResult
  app/services/model_registration.py   MLflow and lakeFS calls
```

Pick and promote models: [docs/promote.md](../../docs/promote.md).
