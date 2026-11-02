# How the steps use datasets and write results

This page is for someone reading or changing the steps. Each step is a folder under `steps/`
that installs kaos-lib (`kaos_lib`, at the pin in `KAOS_LIB_PACKAGE` in its Dockerfile) and copies
only its own `app/`.

## Reading the dataset

| step | what it does with the dataset |
|---|---|
| `config-validation` | parses the pick with `DatasetRef.parse`, resolves it (`datasets.resolve`) and checks the commit has a `data.yaml`; says which versions exist when the pick is wrong |
| `dataset-loading` | resolves the pick **once** to a commit, and outputs it in `manifest-summary` as `dataset_commit` (and `data-yaml`) |
| `model-training` | in the cluster, streams images and labels from the lakeFS S3 gateway at that commit (no full copy); on MeluXina, reads the staged copy through the package (`lakefs_io.materialize`) |
| `qat-finetune`, `model-quantization` | read the dataset as local files for calibration and parity, with `lakefs_io.materialize(dataset_ref, dataset_commit, dest)`, using the `dataset_ref` and `dataset_commit` that training recorded in `training-result` |

`lakefs_io.materialize` is `datasets.materialize` with the ref pinned to the commit
(`DatasetRef.parse(ref).with_commit(commit)`). It downloads `data.yaml`, `images/` and `labels/`
once, or uses the copy the platform staged on MeluXina (`KUBECORE_STAGED_DIR`), and returns a
`data.yaml` whose `path:` points at it.

## Writing results

Training, QAT and quantization write through the package's run writer (`lakefs_io.run_writer`,
`outputs.run`):

1. `model-training` opens the writer for `outputs-<app>`, experiment, and run
   (`KUBECORE_RUN_ID`). If the branch `run-<run>-checkpoints` already holds a `last.pt`, training
   **resumes** from it.
2. Every `checkpoint_interval` epochs the checkpoint is saved on that branch (as `epoch_NNNN.pt` and
   `last.pt`); `best.pt`, `last.pt` and exports are saved at the end. Each save is a commit on the
   run branch only.
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
`training-result` carries `best_checkpoint_s3` (`s3://outputs-<app>/main/checkpoints/<experiment>/<run>/best.pt`),
which the next steps read back with `lakefs_io.download_object`.

## Changing a step

- Keep `lakefs_io.py` identical in the three steps that have it (a test checks it): each image
  carries only its own `app/`.
- Never copy the repository's `kubecore/` folder into a step image: it is the platform's engine,
  with the same import name as the package.
- Step tests live in `step_tests/` and run with the `pytest` command with the package installed
  (DEVELOPER.md §15).
