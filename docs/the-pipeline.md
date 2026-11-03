# What the pipeline does

This page has no tasks. It explains the seven steps, so the run form makes sense. You press
**Submit** once, and the steps run in this order:

| # | step | runs | what it does |
|---|---|---|---|
| 1 | `config-validation` | always | checks your form values fit together, and that the dataset and version you picked exist and have a `data.yaml`. If something is off it stops here, in seconds, before any GPU time is spent |
| 2 | `dataset-loading` | always | turns your `dataset` pick (for example `dataset-cats@v1`) into one lakeFS commit, which every later step reads |
| 3 | `data-summary` | always, alongside training | prints one line about the dataset settings (`📊 Data summary: dataset=... source=... seed=...`) |
| 4 | `model-training` | always | trains the YOLO pose model on a GPU for `train-epochs` epochs, streaming the images from lakeFS. It saves a checkpoint every `train-checkpointing-interval_epochs` epochs, logs to MLflow, and on a retry resumes from its last checkpoint |
| 5 | `qat-finetune` | only when `quantization-mode` = `qat` | goes on training the model in a way that prepares it for INT8 |
| 6 | `model-quantization` | when `quantization-mode` is `ptq` or `qat` | makes an INT8 `.tflite` model and checks it still matches the full model. It fails on purpose if they drift apart by more than `quantization-parity_max_abs_error` |
| 7 | `model-registration` | always | registers the best checkpoint in MLflow, tags its lakeFS commit `model-<name>-v<N>`, and optionally promotes it (`registration-promote_to`) |

With `quantization-mode` = `none`, the default, steps 5 and 6 show as *skipped*. The pipeline
trains a model and registers it. That is the simple path.

## Where it all goes

- **Weights and exports:** `outputs-<app>` in lakeFS, under `checkpoints/<experiment>/<run>/`
  (`best.pt`, `last.pt`, `epoch_NNNN.pt`). The INT8 model goes under
  `quantization/<experiment>/<run>/`.
- **Metrics, parameters, plots:** the project's MLflow, in the experiment named by
  `experiment-name`.
- **The registered model:** MLflow's model registry; `object-pose-yolo` unless you set
  `registration-registered_model_name`.

Next: [Upload images and labels](guide-upload-images.md).
