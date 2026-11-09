# qat-finetune

Quantization-aware training (QAT): a short fine-tune of the trained model that prepares it for INT8, then an INT8 TFLite export.

- **Runs on:** a GPU, only when `quantization-mode` is `qat`.
- **Reads:** `experiment`, `train` (`train-qat-epochs`, `train-qat-lr`) and `quantization` from the form, and `training-result`.
- **Input check:** an empty or unloadable `best_checkpoint_s3` from training is refused with a message that says so.
- **Output:** `qat-result` with the TFLite location, consumed by `model-quantization`.
- **Settings:** [docs/parameters.md](../../docs/parameters.md) (model: `app/models/quantization.py`; `calibration_frames` is at least 100).
- **Dependencies:** the CUDA/QAT stack is pinned in `requirements-cuda.txt` and `requirements-qat.txt`.

- **Resumable:** after each epoch the step saves the model (with its quantization scales), the optimizer and the number of finished epochs into the platform's checkpoint folder (`kaos_lib.checkpoint`, file `qat.bin`; code in `app/services/qat_resume.py`). A repeated step (node lost, time limit) loads it, runs only the remaining epochs and logs into the same MLflow run. See [docs/resilience.md](../../docs/resilience.md).

The step is started by the pipeline; there is no separate command line to run it by hand. `tools/qat_gpu_smoketest.py` checks a GPU machine can run it.
