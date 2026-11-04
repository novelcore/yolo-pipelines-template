# model-quantization

Makes the INT8 TFLite model.

- **Runs on:** CPU, only when `quantization-mode` is `ptq` or `qat`.
- **`ptq`:** exports the trained FP32 `best.pt` to INT8 TFLite after calibrating on `quantization-calibration_frames` training images (at least 100, at most 10000).
- **`qat`:** packages the INT8 TFLite that `qat-finetune` made.
- **Parity:** compares INT8 and FP32 on `quantization-parity_frames` images and logs `parity_max_abs_error` and `parity_passed` to MLflow. This is diagnostic: it does not fail the step.
- **Writes:** the `.tflite` to `outputs-<pipeline>` under `quantization/<experiment>/<run>/`.
- **Output:** `quantization-result`.
- **Settings:** [docs/parameters.md](../../docs/parameters.md) (model: `app/models/quantization.py`).

The step is started by the pipeline with the composed params and the training result; there is no separate command line to run it by hand.
