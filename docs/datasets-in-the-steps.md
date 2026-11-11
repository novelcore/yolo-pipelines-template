# Under the hood: how the steps use datasets and write results

*Optional.* You do not need this page to train. It is for someone reading or changing the steps. Each step is a folder under `steps/`
that installs kaos-lib (`kaos_lib`, at the pin in `KAOS_LIB_PACKAGE` in its Dockerfile) and copies
only its own `app/`.

## Reading the dataset

| step | what it does with the dataset |
|---|---|
| `config-validation` | does not read the dataset |
| `model-training` | at its start resolves the pick with `datasets.resolve`, which returns the one commit the platform pinned for the run (the run's data lock, `KAOS_LIB_LOCK`), and reads the dataset's `data.yaml` at that commit. For your own dataset in the cluster, streams images and labels from the project's lakeFS S3 gateway at that commit (no full copy); for a **shared** dataset (`<owner>/dataset-x@v1`, which lives in the owner's lakeFS) and on Slurm, materializes the data locally first through the package (`lakefs_io.materialize`) |
| `qat-finetune`, `model-quantization` | read the dataset as local files for calibration and parity, with `lakefs_io.materialize(dataset_ref, dataset_commit, dest)`, using the `dataset_ref` and `dataset_commit` that training recorded in `training-result` |

`lakefs_io.materialize` is `datasets.materialize` with the ref pinned to the commit
(`DatasetRef.parse(ref).with_commit(commit)`). It downloads `data.yaml`, `images/` and `labels/`
once, or uses the copy staged on the Slurm machine, and returns a
`data.yaml` whose `path:` points at it.

## Writing results

Training, QAT and quantization write through the package's run writer (`lakefs_io.run_writer`,
`outputs.run`):

1. `model-training` opens the writer for `outputs-<pipeline>`, experiment, and run
   (the run's id). Training never resumes from this branch: resuming is the platform's checkpoint
   folder (`KAOS_CHECKPOINT_DIR`), see [When a machine is lost](resilience.md).
2. `best.pt`, `last.pt` and exports are saved at the end. Each save is a commit on the run branch only.
3. `finish` merges the branch into `main`: **one commit**, `checkpoints: <experiment>/<run>`, with
   `dataset_ref`, `dataset_commit` and `mlflow_run_id` in its details, and deletes the branch.
4. QAT and quantization do the same with `kind="quantization"` for the INT8 `.tflite`.
5. Training then tags its MLflow run with `kaos.outputs_repo`, `kaos.outputs_commit` (the merge
   commit), `kaos.outputs_path` and `kaos.workflow`. `model-registration` reads them back, creates
   the lakeFS tag `model-<slug(name)>-v<N>` on that commit for the `best.pt` version it registers
   (an existing tag on the same commit is fine; on another commit the step fails), and sets
   `kaos.outputs_repo`, `kaos.outputs_commit`, `kaos.lakefs_tag` and `kaos.workflow` on the
   model version.

If a step raises, the branch stays, nothing reaches `main`, and the retry continues from the branch.
`training-result` carries `best_checkpoint_s3` (`s3://outputs-<pipeline>/main/checkpoints/<experiment>/<run>/best.pt`),
which the next steps read back with `lakefs_io.download_object`. It is never empty: when a run has no `best.pt`
(a continued run restores it through kaos-lib; without it Ultralytics writes none if no epoch after the restart beat the
earlier best), `best_checkpoint_s3` is the `last.pt` location, and a run with no weights at all fails. `qat-finetune` and
`model-quantization` refuse an empty checkpoint with a message that says so.

## Models in MLflow

Training also logs `best.pt` as an MLflow 3 **logged model** `<experiment>-best` (the
experiment's **Models** tab) with `mlflow.create_external_model`: the weights stay in lakeFS,
MLflow holds only a metadata `MLmodel`. Its source run is the training run; its params are
`epochs`, `imgsz` and `model_variant`; its tags are the four `kaos.*` above plus
`kaos.dataset_ref`; the final val metrics (`val/*`, from the trainer's last validation of
`best.pt`) are logged against it. The run gets `kaos.logged_model_id`. A retried step reuses the
run's logged model instead of making a second one, and a failure only warns: the results are
already in lakeFS and registration falls back to the lakeFS path.

When `model-registration` registers version N of `best.pt` as `<name>`, it registers it **from
the logged model** (`models:/<model_id>`), so the version links to the training run and to the
logged model; it tags the commit in lakeFS as **`model-<name>-v<N>`** (characters other than
`A-Za-z0-9_-` become `-`) and sets `kaos.outputs_repo`, `kaos.outputs_commit`, `kaos.lakefs_tag`
and `kaos.workflow` on the model version. A run with no `kaos.logged_model_id` (older runs, or
training could not log it) is registered from the `s3://` lakeFS path of `best.pt`, with a
warning. `last.pt` is registered from its lakeFS path, without a tag. A run with no
`kaos.outputs_commit` (older runs) is registered without a lakeFS tag. The INT8 models of
`model-quantization` and `qat-finetune` are not logged models.

## Changing a step

- Keep `lakefs_io.py` identical in the three steps that have it (a test checks it): each image
  carries only its own `app/`.
- Step tests live in `step_tests/` and run with `python -m pytest` with the package installed
  (DEVELOPER.md §15).
