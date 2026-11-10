# config-validation

The first step. It checks the run form before any training time is spent, and stops with a message that names what to fix.

- **Checks:** that `model-pretrained_weights` and the `train-checkpointing-resume_from` checkpoint exist when set. It does not check the dataset: the platform pins the run's datasets before the first step, and `model-training` reads them. A check that must run and finds no credentials fails the step; on a laptop (no platform environment at all) the object-store checks are skipped, and the log says so.
- **Real messages and their fix:** [docs/troubleshooting.md](../../docs/troubleshooting.md).
- **Also checks:** when `quantization-mode` is `ptq` or `qat`, that `quantization-calibration_frames` is 100 to 10000 (the bounds of the INT8 steps), so a bad value fails here and not after training.
