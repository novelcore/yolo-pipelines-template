# config-validation

A gate before any machine is paid for: it stops the run when a setting would be rejected later.

- **Runs on:** CPU, first.
- **Reads:** the settings `experiment`, `model`, `quantization` and `registration`.
- **Checks:** that `quantization-calibration_frames` is from 100 to 10000 when `quantization-mode` is `ptq` or `qat`, and that `model-pretrained_weights`, when set, is a `<repository>@<version>/<path>` that exists. That `experiment-name` and `registration-registered_model_name` hold only letters, digits, `-` and `_`, because the project's MLflow accepts only an experiment or a model named `<app>` or `<app>.<name>` (`app/names.py`). It does not check the dataset or MLflow.
- **Hands on:** `checked`.
