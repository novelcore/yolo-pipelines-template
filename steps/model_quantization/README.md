# model-quantization

Makes the INT8 `.tflite` of the trained model and checks it against the full model.

- **Runs on:** CPU, when `quantization-mode` is `ptq` or `qat`.
- **Reads:** the setting `quantization`; the model `checkpoints` and the dataset `prepared` with `step.input(...)`; in `qat` mode also the model `qat` that `qat-finetune` made.
- **Writes:** the model `quantization`: the INT8 `.tflite` (exported here for `ptq`, the one of `qat-finetune` checked and saved for `qat`), under `quantization/<experiment>/<run>/`.
- **Hands on:** the MLflow run, the file name, the parity result (`parity_passed`, `parity_max_abs_error`).
