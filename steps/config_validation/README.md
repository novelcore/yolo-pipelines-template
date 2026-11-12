# config-validation

A gate before any machine is paid for: it stops the run when a setting would be rejected later.

- **Runs on:** CPU, first.
- **Reads:** the settings `model` and `quantization`.
- **Checks:** that `quantization-calibration_frames` is from 100 to 10000 when `quantization-mode` is `ptq` or `qat`, and that `model-pretrained_weights`, when set, is a `<repository>@<version>/<path>` that exists. It does not check the dataset or MLflow.
- **Hands on:** `checked`.
