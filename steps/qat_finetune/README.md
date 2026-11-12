# qat-finetune

Fine-tunes the trained model with quantization-aware training and exports an INT8 `.tflite`.

- **Runs on:** a GPU, only when `quantization-mode` = `qat`.
- **Reads:** the settings `train` (`train-qat-epochs`, `train-qat-lr`) and `quantization`; the model `checkpoints` and the dataset `prepared` (calibration) with `step.input(...)`; the MLflow run from the result of `model-training`.
- **Writes:** the model `qat`: the INT8 `.tflite`, saved under `qat/<experiment>/<run>/`. A step run again continues from the last finished epoch of its checkpoint folder (`app/services/qat_resume.py`).
- **Hands on:** the MLflow run, the file name, the parity result.
