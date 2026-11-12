# Under the hood: how the steps use datasets and write results

*Optional.* You do not need this page to train. It is for someone reading or changing the steps. Each step is a folder under `steps/`
that installs kaos-lib (`kaos_lib`, at the pin in `KAOS_LIB_PACKAGE` in its Dockerfile) and copies only its own `app/`. A step
gets everything it reads and hands on everything it writes through [the step API](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/step.html):
no step reads a platform variable, parses a dataset reference or relays a path.

## What each step reads and writes

| step | reads | writes | hands on |
|---|---|---|---|
| `config-validation` | the settings `model` and `quantization` | nothing | that it checked |
| `prepare-data` | `step.input("dataset")`: the version of your `dataset` pick that the platform pinned for the run | the dataset `prepared` (`step.output("prepared")`) | the images kept per split |
| `model-training` | `step.input("prepared")`: exactly the version `prepare-data` wrote | the model `checkpoints`: `best.pt`, `last.pt`, exports | the MLflow run, the checkpoint's file name, the validation metrics |
| `qat-finetune` | the model `checkpoints` and the dataset `prepared` | the model `qat`: the INT8 `.tflite` | the MLflow run, the file name |
| `model-quantization` | the model `checkpoints` (and `qat`) and the dataset `prepared` | the model `quantization`: the INT8 `.tflite` | the MLflow run, the file name |
| `model-registration` | the results of the steps before it, and where they were saved | nothing: it registers in MLflow | the registered versions |

`pipeline.py` declares each line of this table (`inputs=`, `needs=`, `produces=`); `kaosctl lint` checks the declarations
against the code.

## Reading the dataset

`step.input("dataset")` is a folder with the pinned version of the dataset: downloaded once and kept, or, on HPC (Slurm),
the copy the platform staged next to the job. The folder holds only the dataset's own files and is never written to. It is
the same for your own datasets and for a dataset shared with you.

`prepare-data` reads it. `model-training`, `qat-finetune` and `model-quantization` read the dataset `prepared` that
`prepare-data` wrote in the same run, so all of them see the same files. Training copies the folder's `data.yaml` into a
temporary file with its `path:` set to the folder, and Ultralytics reads the images and labels from the files. There is no
streaming of images from the store while it trains: the files are on the machine first.

## Writing results

A step writes into the folder `step.output("<name>")` and the step API saves it when the step ends (`step.result(...)`):

| declared in `pipeline.py` | saved as |
|---|---|
| `model("checkpoints")`, `model("qat")`, `model("quantization")` | the files of this run, in `outputs-<pipeline>` under `<name>/<experiment>/<run>/`, one commit per output and run |
| `dataset("prepared")` | a new version of the dataset `dataset-<your app>-prepared` |

Nothing reaches `outputs-<pipeline>` until the step has ended well, and a step that is run again writes exactly what the new
attempt wrote. What the step hands on in `step.result(...)` is small (at most 8 KiB): the MLflow run, file names and metrics.
The step API adds what it read and what it wrote, each with its exact commit, to the result of the step, so the next step
knows where to read.

## Models in MLflow

`model-registration` shows `best.pt` in the experiment's **Models** tab as an MLflow 3 **logged model** `<experiment>-best`
(`mlflow.create_external_model`): the weights stay in lakeFS, MLflow holds only metadata. Its source run is the training run; its
params are `epochs`, `imgsz` and `model_variant`; its tags are `kaos.outputs_repo`, `kaos.outputs_commit` (the commit that holds the
weights), `kaos.outputs_path`, `kaos.workflow` and `kaos.dataset_ref`; the final validation metrics of the training run are logged
against it. The training run gets the same tags and `kaos.logged_model_id`. A step that is run again reuses the run's logged model.

It then registers version N of `best.pt` as `<name>` **from the logged model** (`models:/<model_id>`), tags the commit in lakeFS as
**`model-<name>-v<N>`** (characters other than `A-Za-z0-9_-` become `-`; an existing tag on the same commit is fine, on another commit
the step fails), and sets `kaos.outputs_repo`, `kaos.outputs_commit`, `kaos.lakefs_tag` and `kaos.workflow` on the model version.
`last.pt` is registered from its path in lakeFS, pinned to the commit, without a lakeFS tag. The INT8 models are not logged models;
their location is a tag of the version, `export_int8_tflite`. The version also says where the data came from: `dataset_version`
and `dataset_commit` (the dataset you picked), `prepared_dataset` (the version training read) and `dataset_sample_size`.

If MLflow cannot log the model, the step fails and says so: a model that cannot be linked to its run is not registered.

## Changing a step

- Step tests live in `step_tests/` and run with `python -m pytest` with kaos-lib installed (`step_tests/README.md`).
- Add a step or a setting: [Change the pipeline](change-the-pipeline.md).
