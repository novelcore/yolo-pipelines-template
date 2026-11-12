# What the pipeline does

This page explains the six steps, so the run form makes sense. You submit the run once, and the steps run in this order:

| # | step | runs | what it does |
|---|---|---|---|
| 1 | `config-validation` | always | checks your form values fit together (the INT8 settings, and that a pretrained-weights path exists). If something is off it stops here, in seconds, before any machine is paid for |
| 2 | `prepare-data` | always | reads the one version of your `dataset` pick (for example `dataset-cats@v1`) that the platform pinned when the run started, and writes it as the dataset `prepared`: every image, or, with `data-sample_size`, that many images per split (with their labels), picked by `data-seed`. [Details](#prepare-data-the-first-step-writes-a-dataset) |
| 3 | `model-training` | always | trains the YOLO pose model on a GPU for `train-epochs` epochs on exactly the version `prepare-data` wrote. It logs to MLflow (a checkpoint every `train-checkpointing-interval_epochs` epochs), and when run again (a retry, a lost machine, the time limit) resumes from its last checkpoint |
| 4 | `qat-finetune` | only when `quantization-mode` = `qat` | goes on training the model in a way that prepares it for INT8 |
| 5 | `model-quantization` | when `quantization-mode` is `ptq` or `qat` | makes an INT8 `.tflite` model and compares it with the full model on `quantization-parity_frames` images. The result is logged to MLflow (`parity_passed`, `parity_max_abs_error`); it does not stop the run, so look at it before you use the INT8 file |
| 6 | `model-registration` | always | registers the best checkpoint in MLflow, and optionally promotes it (`registration-promote_to`) |

With `quantization-mode` = `none`, the default, steps 4 and 5 show as *skipped*. The pipeline
prepares the data, trains a model and registers it. That is the simple path.

## `prepare-data`: the first step writes a dataset

`prepare-data` is the step that writes the dataset every later step reads
([Produce a dataset](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/produce-a-dataset.html) explains how a step does that).

- **Without `data-sample_size`** (the default) it keeps every file of your dataset, unchanged.
- **With `data-sample_size` = `50`** it keeps 50 images of each split (`train`, `val`, `test`) and their labels, to try a run
  quickly. `data-seed` picks which: the same seed keeps the same images, another seed other ones. A split with fewer images keeps
  them all. `data.yaml` and every file outside `images/` and `labels/` are kept as they are.
- **Where it goes:** the dataset `dataset-<your app>-prepared`, a dataset of its own in your project ([Datasets](datasets.md)).
  The steps after `prepare-data` read this run's version, so a run is never changed by another run that finishes later.
- **Why a version for a quick test?** A run records exactly which files it trained on. Two runs with the same `dataset`,
  `data-sample_size` and `data-seed` train on the same bytes.

## When a step is run again

A step whose machine is lost, or whose time limit is reached, is run again and continues from its checkpoint
([Long jobs](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/long-jobs.html)). What each step of this pipeline keeps:

| step | kept | when it is run again |
|---|---|---|
| `prepare-data` | nothing: it is quick | it simply runs again |
| `model-training` | `last.pt` at the end of each epoch, and `best.pt` once one exists | training continues with the same total epochs, in the same MLflow run; `best.pt` is put back, so the best checkpoint is never lost. If the run still ends with no `best.pt`, `last.pt` is handed to the next steps; a run with no weights at all fails |
| `qat-finetune` | the model, the optimizer and the number of finished epochs, at the end of each epoch | the fine-tune continues from the last finished epoch, with the same total epochs, in the same MLflow run |
| `model-quantization` | nothing: it is one short export | it simply runs again |

`qat-finetune` and `model-quantization` refuse an empty or unloadable checkpoint with a message that says so.

## Where it all goes

- **Weights and exports:** the files of the run, in the folder `checkpoints/<experiment>/<run>/` (`best.pt`,
  `last.pt`). The INT8 model is in `quantization/<experiment>/<run>/`, and, for `qat`, the model `qat-finetune` made is in
  `qat/<experiment>/<run>/`. [See your results](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/see-your-results.html#the-files-of-a-run) shows how to download them.
- **The prepared data:** the dataset `dataset-<your app>-prepared` ([Datasets](datasets.md)).
- **Metrics, parameters, plots:** the project's MLflow, in the experiment named by
  `experiment-name`.
- **The registered model:** MLflow's model registry; `object-pose-yolo` unless you set
  `registration-registered_model_name`. Set `registration-promote_to` to `champion` or `challenger` on the run form to give the new version that label when the run ends; empty only saves it.

Next: [Datasets](datasets.md).
