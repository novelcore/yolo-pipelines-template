# What the pipeline does

This page has no tasks. It explains the six steps, so the run form makes sense. You press
**Submit** once, and the steps run in this order:

| # | step | runs | what it does |
|---|---|---|---|
| 1 | `config-validation` | always | checks your form values fit together (the INT8 settings, and that a pretrained-weights path exists). If something is off it stops here, in seconds, before any machine is paid for |
| 2 | `prepare-data` | always | reads the one version of your `dataset` pick (for example `dataset-cats@v1`) that the platform pinned when the run started, and writes it as the dataset `prepared`: every image, or, with `data-sample_size`, that many images per split (with their labels), picked by `data-seed`. [Details](#prepare-data-the-first-step-writes-a-dataset) |
| 3 | `model-training` | always | trains the YOLO pose model on a GPU for `train-epochs` epochs on exactly the version `prepare-data` wrote. It logs to MLflow (a checkpoint every `train-checkpointing-interval_epochs` epochs), and when run again (a retry, a lost machine, an HPC time limit) resumes from its last checkpoint |
| 4 | `qat-finetune` | only when `quantization-mode` = `qat` | goes on training the model in a way that prepares it for INT8 |
| 5 | `model-quantization` | when `quantization-mode` is `ptq` or `qat` | makes an INT8 `.tflite` model and compares it with the full model on `quantization-parity_frames` images. The result is logged to MLflow (`parity_passed`, `parity_max_abs_error`); it does not stop the run, so look at it before you use the INT8 file |
| 6 | `model-registration` | always | registers the best checkpoint in MLflow, tags its lakeFS commit `model-<name>-v<N>`, and optionally promotes it (`registration-promote_to`) |

With `quantization-mode` = `none`, the default, steps 4 and 5 show as *skipped*. The pipeline
prepares the data, trains a model and registers it. That is the simple path.

## `prepare-data`: the first step writes a dataset

A step can write a dataset, and the steps after it read exactly that version. `prepare-data` is the first one that does.

- **Without `data-sample_size`** (the default) it keeps every file of your dataset, unchanged. The files are not copied when the
  machine can share them, and a run that writes the same files as the last one adds no version and uploads nothing.
- **With `data-sample_size` = `50`** it keeps 50 images of each split (`train`, `val`, `test`) and their labels, to try a run
  quickly. `data-seed` picks which: the same seed keeps the same images, another seed other ones. A split with fewer images keeps
  them all. `data.yaml` and every file outside `images/` and `labels/` are kept as they are.
- **Where it goes:** the dataset `dataset-<pipeline>-prepared`, a repository of its own in your project. A run that writes new
  files seals a new version; one that writes files an existing version already has adds none. The dataset's `latest` moves to this run's version only when you ask: `plan: {promote: {prepared: true}}` in the run's
  advanced settings. The steps after `prepare-data` read this run's version either way, so a run is never changed by another
  run that finishes later.
- **Why a version for a quick test?** A run records exactly which files it trained on. Two runs with the same `dataset`,
  `data-sample_size` and `data-seed` train on the same bytes.

The step runs on the platform's own machines: a step that writes a dataset is not offered an HPC (Slurm) class yet. Training
and the INT8 steps can still run on HPC.

## When a step is run again

A step whose machine is lost, or whose HPC (Slurm) time limit is reached, is run again by the platform and continues
from its checkpoint. How that works, the settings and the log messages are in the kaos-lib guide:
[Long runs: checkpoints](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/checkpoint.html). What each step of this pipeline keeps:

| step | kept | when it is run again |
|---|---|---|
| `prepare-data` | nothing: it is quick | it simply runs again |
| `model-training` | `last.pt` at the end of each epoch, and `best.pt` once one exists | training continues with the same total epochs, in the same MLflow run; `best.pt` is put back, so the best checkpoint is never lost. If the run still ends with no `best.pt`, `last.pt` is handed to the next steps; a run with no weights at all fails |
| `qat-finetune` | the model, the optimizer and the number of finished epochs, at the end of each epoch | the fine-tune continues from the last finished epoch, with the same total epochs, in the same MLflow run |
| `model-quantization` | nothing: it is one short export | it simply runs again |

`qat-finetune` and `model-quantization` refuse an empty or unloadable checkpoint with a message that says so.

A step that ends without handing on its result fails, and the steps after it do not start: no step ever gets an empty result
from an earlier one.

## Where it all goes

- **Weights and exports:** `outputs-<pipeline>` in lakeFS, under `checkpoints/<experiment>/<run>/`
  (`best.pt`, `last.pt`). The INT8 model goes under `quantization/<experiment>/<run>/`, and, for `qat`, the model
  `qat-finetune` made under `qat/<experiment>/<run>/`.
- **The prepared data:** the dataset `dataset-<pipeline>-prepared` ([Datasets](datasets.md)).
- **Metrics, parameters, plots:** the project's MLflow, in the experiment named by
  `experiment-name`.
- **The registered model:** MLflow's model registry; `object-pose-yolo` unless you set
  `registration-registered_model_name`.

Next: [Upload images and labels](prepare-dataset.md).
