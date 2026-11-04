# qat-finetune

Quantization-aware training (QAT): a short fine-tune of the trained model that prepares it for INT8, then an INT8 TFLite export.

- **Runs on:** a GPU, only when `quantization-mode` is `qat`.
- **Reads:** `experiment`, `train` (`train-qat-epochs`, `train-qat-lr`) and `quantization` from the form, and `training-result`.
- **Output:** `qat-result` with the TFLite location, consumed by `model-quantization`.
- **Settings:** [docs/parameters.md](../../docs/parameters.md) (model: `app/models/quantization.py`; `calibration_frames` is at least 100).
- **Dependencies:** the CUDA/QAT stack is pinned in `requirements-cuda.txt` and `requirements-qat.txt`.

The step is started by the pipeline; there is no separate command line to run it by hand. `tools/qat_gpu_smoketest.py` checks a GPU machine can run it.
