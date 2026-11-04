# config-validation

The first step. It checks the run form before any training time is spent, and stops with a message that names what to fix.

- **Checks:** that the picked `dataset` (`<dataset>@<version>`, or `<owner>/<dataset>@<version>`) and version exist and that the commit has a `data.yaml` at its root; that `model-pretrained_weights` and a `resume_from` checkpoint exist when set; that MLflow is reachable.
- **Real messages and their fix:** [docs/troubleshooting.md](../../docs/troubleshooting.md).
- **Also checks:** when `quantization-mode` is `ptq` or `qat`, that `quantization-calibration_frames` is 100 to 10000 (the bounds of the INT8 steps), so a bad value fails here and not after training.
