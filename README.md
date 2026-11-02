# yolo-pipelines-template

**The official YOLO pipelines KubeApp template.** Scaffolding a KubeApp with
`kubeAppTemplateRef: yolo-pipelines` gives you a complete, working YOLO pose
pipeline: dataset loading, real YOLOv8 training with authenticated MLflow
tracking, optional QAT fine-tune (GPU or CPU-routed), INT8 TFLite export (PTQ or
QAT), and model registration. Results go to the app's own `outputs-<app>` lakeFS
repository.

It is the worked example built on the scaffold
[`hera-pipeline-template`](https://github.com/novelcore/hera-pipeline-template): start
from the scaffold to write your own pipeline, read this one to see a real one. The
platform injects the environment's MLflow, lakeFS and compute context at render time
(each app binds to one PRD-858 `ml`-type environment via `environmentSelector`);
nothing environment-specific lives in this repo.

## Contents

- [The pipeline](#the-pipeline)
- [Datasets](#datasets)
- [Results](#results)
- [kaos-lib, the platform library](#kaos-lib-the-platform-library)
- [Running steps on MeluXina](#running-steps-on-meluxina-hpc)
- [Local iteration](#local-iteration-no-cluster)
- [Where to read next](#where-to-read-next)

## The pipeline

```python
# pipeline.py — the ENTIRE pipeline definition
with pipeline("ml-pipeline") as p:  # platform renames to {app}-pipeline at release
    validate = step("config-validation", reads=["data", "dataset", "model"])
    load = step("dataset-loading", reads=["data", "dataset"], needs=[validate],
                outputs=["data-yaml", "manifest-summary"])
    summary = step("data-summary", reads=["data", "dataset", "summary"], needs=[load])
    train = step("model-training", gpu=True, needs=[load], hpc_time_limit="12h",
                 reads=["experiment", "dataset", "data", "model", "train", "image_processing", "logging"],
                 outputs=["training-result"])
    qat = step("qat-finetune", gpu=True, needs=[train], disk="20Gi",
               reads=["experiment", "train", "quantization"],
               when="{{workflow.parameters.quantization-mode}} == qat",
               outputs=["qat-result"])
    quant = step("model-quantization", needs=[train, qat],
                 reads=["experiment", "quantization"],
                 when="{{workflow.parameters.quantization-mode}} != none",
                 outputs=["quantization-result"])
    register = step("model-registration", needs=[train, quant],
                    reads=["dataset", "data", "model", "registration"],
                    outputs=["registration-result"])
```

**One rule: the `config/` tree IS the submit form.** Every scalar in `config/`
becomes a form field; every group directory becomes a dropdown. Add a leaf → a
field appears. No parameter wiring anywhere.

```
pipeline.py            the DAG (steps, reads=, gpu=, needs=, when=)
config/                THE experiment config tree — all your parameters
  config.yaml            the defaults list (which group options are default)
  dataset.yaml           the `dataset` field: <dataset-repo>@<version>
  data.yaml train.yaml … sections of scalar leaves
  train/optimizer/*.yaml a config GROUP → a dropdown
steps/<name>/          step code + Dockerfile (one dir per step)
kubecore/              platform-owned engine — SEEDED, DO NOT EDIT
tests/, step_tests/    engine tests; step-code tests (two commands, see DEVELOPER.md §15)
docs/                  the user guide (see "Where to read next")
```

## Datasets

A **dataset is a lakeFS repository** named `dataset-<name>`, with the YOLO layout at its
root: `data.yaml`, `images/{train,val,test}/`, `labels/{train,val,test}/`. The project's
default is `dataset-default`.

- **Versions** are tags on `main`: `v1`, `v2`, … `latest` is the head of `main`. A run
  picks one and every step reads that same commit, whatever happens to the dataset meanwhile.
- **Changes** go in through a **pull request**; after a merge, `kaos-cli dataset release`
  tags the next version.
- **Pick it in the pipeline.** The run form's `dataset` field is a dropdown of
  `<dataset-repo>@<version>` choices (default `dataset-default@latest`, or its newest
  version). `config-validation` checks the pick exists and has a `data.yaml`;
  `dataset-loading` pins its commit; later steps read that commit.
- **Pose datasets.** This pipeline trains pose: `data.yaml` carries `kpt_shape` (and
  `flip_idx`), labels carry the keypoints. `kaos-cli dataset validate`/`push` check pose,
  segmentation and detection labels.
- **Templates are optional.** `kaos-cli dataset create` can start a repository from the
  `ultralytics` template, but any `dataset-*` repository made by hand becomes a dataset after
  its first commit (`--template none` makes an empty one). A repository with no template
  label validates nothing on `push`; pass `--format ultralytics` to get the YOLO checks.
  Any repository can be read from code: `datasets.materialize("<repo>@<ref>")`.
- A pipeline **never writes to a dataset**.

Every command, from an empty project to a finished run:
[docs/first-run.md](docs/first-run.md). How the steps read the data:
[docs/datasets-in-the-steps.md](docs/datasets-in-the-steps.md).

## Results

Every step that produces files writes them to the app's repository **`outputs-<app>`**, on
`main`, as **one merge commit per step per run**, naming the run:

| step | where | what |
|---|---|---|
| `model-training` | `checkpoints/<experiment>/<run>/` | `best.pt`, `last.pt`, `epoch_NNNN.pt`, exports |
| `qat-finetune`, `model-quantization` | `quantization/<experiment>/<run>/` | the INT8 `.tflite` |

The step saves each checkpoint on its own branch `run-<run>-<kind>` as it appears, and merges
at the end, so a lost pod costs nothing: **a retried training step resumes** from the last
saved checkpoint instead of starting over. The merge commit records the app, experiment, run,
the dataset (`dataset_ref`, `dataset_commit`) and the MLflow run id. Metrics, plots and the
registered model are in MLflow ([docs/results.md](docs/results.md)).

**The model is linked both ways.** The model is the `outputs-<app>` repository at the training
merge commit. Training sets these tags on its MLflow run (named after the workflow):
`kaos.outputs_repo` (`outputs-<app>`), `kaos.outputs_commit` (the merge commit on `main`),
`kaos.outputs_path` (`checkpoints/<experiment>/<run>/`) and `kaos.workflow`.

Training also logs `best.pt` as an MLflow 3 **logged model** `<experiment>-best` (the
experiment's **Models** tab) with `mlflow.create_external_model`: the weights stay in lakeFS,
MLflow holds only a metadata `MLmodel`. Its source run is the training run; its params are
`epochs`, `imgsz` and `model_variant`; its tags are the four `kaos.*` above plus
`kaos.dataset_ref`; the final val metrics (`val/*`, from Ultralytics' last validation of
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

## kaos-lib, the platform library

The steps use [kaos-lib](https://github.com/kaos-io/kaos-lib) (`kaos_lib`, formerly the `kubecore`
package), pinned in every step Dockerfile as `KAOS_LIB_PACKAGE`: dataset refs and their commits,
the local or MeluXina-staged copy of a dataset, the run writer for results, and the
`kaos-cli` command. Install it on your computer inside a virtual environment:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install "kaos-lib[zitadel] @ https://github.com/kaos-io/kaos-lib/archive/<commit>.zip"
kaos-cli login                    # opens your browser once
```

Run the commands inside your app folder: it carries the file that tells the tool where your
project's lakeFS is.

```bash
kaos-cli dataset create <name> --classes a,b [--template ultralytics|none]
kaos-cli dataset pull <repo>@<ref> [dir] [--prefix folder/]
kaos-cli dataset validate <dir> --format ultralytics
kaos-cli dataset push <dir> <repo> -m "message" [--format ultralytics]
kaos-cli dataset release <repo>
kaos-cli dataset list ; kaos-cli dataset versions <repo>
```

From code: `datasets.materialize("dataset-cats@v2")` returns the local `data.yaml`
(a folder for a repository with no template label); `outputs.run(app, experiment, run)` is
the run writer.

The package and the vendored `kubecore/` engine share an import name and never share a
`sys.path`: step images install the package and copy only `app/`. That is why engine tests
and step tests are two commands (DEVELOPER.md §15).

## Running steps on MeluXina (HPC)

When the project's pool is HPC-enabled, every step's `{step}-class` dropdown on the
Argo submit form also offers the MeluXina classes:

| class | what it is | typical use |
|---|---|---|
| `meluxina-gpu` | 4× A100-40GB, 2× EPYC 7452, 512 GB | training, QAT fine-tune |
| `meluxina-cpu` | 2× EPYC 7452 (128 cores), 512 GB | heavy preprocessing, PTQ export |
| `meluxina-largemem` | 4 TB RAM | datasets that do not fit a node |

Pick one for a step and only that step runs as a Slurm job on MeluXina; every other
step keeps its in-cluster class — mixing (training on `meluxina-gpu`, quantization on
`gpu-small`, registration in-cluster) is the normal case. Nothing in this repository changes:

- **The dataset is staged** on MeluXina before the job starts, by repository and commit (a new
  version copies only the files that changed), and the step reads it through the same
  `datasets.materialize(...)` call as in the cluster.
- **Results come back the same way**: the step runs your image and writes checkpoints with the
  run writer, so they appear in `outputs-<app>` as for an in-cluster run; the step's declared
  outputs are brought back so the next step consumes them as usual.
- Queue time is real (minutes to hours) — the pipeline waits for you; the
  `pipeline-info` field on the form lists the classes and the account.
- Give long steps a wall-clock limit in `pipeline.py`: `step(..., hpc_time_limit="12h")`
  (default 4 h). See [DEVELOPER.md §8.1](DEVELOPER.md).
- Steps pinned with a `compute-class` annotation stay in-cluster.

## Local iteration (no cluster)

```bash
./run.sh                                              # venv + render + enhance + compose
python -m kubecore.compose train.epochs=5 train/optimizer=adamw   # try overrides locally
```

`out/params.yaml` is exactly what your steps receive at run time; a typo or a
bad `reads=` fails locally with the same message the cluster gives you.

## What the platform does for you (you never write this)

Image supply-chain (Zot registry), MLflow/lakeFS/outputs env + secrets,
per-step compute-class selection + node scheduling from your KubePool's classes,
per-run sizing knobs, `/dev/shm`, GitOps release, and the Argo submit form —
all injected at release time. Your `pipeline.py` stays pure structure.

## Where to read next

| you want to | read |
|---|---|
| train for the first time: dataset, run, results | [docs/first-run.md](docs/first-run.md) |
| understand datasets, versions, templates | [docs/datasets.md](docs/datasets.md) |
| see how the steps read data and write results | [docs/datasets-in-the-steps.md](docs/datasets-in-the-steps.md) |
| add your first step | [ADD-A-STEP.md](ADD-A-STEP.md) |
| the complete operating manual (config, tests, HPC) | [DEVELOPER.md](DEVELOPER.md) |
| how the platform works internally | [MECHANISMS.md](MECHANISMS.md) |

## License

This project is licensed under the Apache License, Version 2.0. See the
[LICENSE](LICENSE) file for the full terms.
