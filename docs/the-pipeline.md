# What the pipeline does

This page has no tasks. It explains the five steps, so the run form makes sense. You press
**Submit** once, and the steps run in this order:

| # | step | runs | what it does |
|---|---|---|---|
| 1 | `config-validation` | always | checks your form values fit together (the INT8 settings, and that a pretrained-weights path or a checkpoint to resume from exists). If something is off it stops here, in seconds, before any GPU time is spent |
| 2 | `model-training` | always | trains the YOLO pose model on a GPU for `train-epochs` epochs on the one commit of your `dataset` pick (for example `dataset-cats@v1`) that the platform pinned when the run started, streaming the images from lakeFS. It logs to MLflow (a checkpoint every `train-checkpointing-interval_epochs` epochs), and when run again (a retry, a lost machine, an HPC time limit) resumes from its last checkpoint |
| 3 | `qat-finetune` | only when `quantization-mode` = `qat` | goes on training the model in a way that prepares it for INT8 |
| 4 | `model-quantization` | when `quantization-mode` is `ptq` or `qat` | makes an INT8 `.tflite` model and compares it with the full model on `quantization-parity_frames` images. The result is logged to MLflow (`parity_passed`, `parity_max_abs_error`); it does not stop the run, so look at it before you use the INT8 file |
| 5 | `model-registration` | always | registers the best checkpoint in MLflow, tags its lakeFS commit `model-<name>-v<N>`, and optionally promotes it (`registration-promote_to`) |

With `quantization-mode` = `none`, the default, steps 3 and 4 show as *skipped*. The pipeline
trains a model and registers it. That is the simple path.

## Where it all goes

- **Weights and exports:** `outputs-<pipeline>` in lakeFS, under `checkpoints/<experiment>/<run>/`
  (`best.pt`, `last.pt`). The INT8 model goes under
  `quantization/<experiment>/<run>/`.
- **Metrics, parameters, plots:** the project's MLflow, in the experiment named by
  `experiment-name`.
- **The registered model:** MLflow's model registry; `object-pose-yolo` unless you set
  `registration-registered_model_name`.

Next: [Upload images and labels](prepare-dataset.md).
