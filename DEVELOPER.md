# Pipeline Developer Guide

The operating manual for authoring and running your ML pipeline. You need
Python and YAML. You do not need Kubernetes — anything you don't see in
this guide is handled for you.

---

## 1. Mental model

Your repo is:

```
your-repo/
├── pipeline.py              # the DAG: steps, what they read, how they connect
├── config/                  # THE experiment config tree (all your parameters)
│   ├── config.yaml          #   defaults list (which group options are default)
│   ├── dataset.yaml         #   the `dataset` field: <dataset-repo>@<version>
│   ├── data.yaml  train.yaml  quantization.yaml  ...
│   ├── train/optimizer/{sgd,adam,adamw}.yaml     # a config GROUP
│   └── image_processing/{mosaic_default,...}.yaml
├── step_tests/              # tests of your step code (§15)
└── steps/
    ├── model_training/      # Dockerfile + entry point (your code)
    └── ...
```

**One rule to remember: the config tree IS the submit form.** Every
scalar value in `config/` becomes a form field automatically. Every
group directory becomes a dropdown. Add a leaf → a field appears.
There is no parameter wiring anywhere else.

**What happens on push:** push to `dev` → the platform builds your step
images and releases a new pipeline version (~minutes).

**What happens on submit:** the values you set on the run form are
composed with your tree into ONE resolved `params.yaml` — by the
pipeline's first step, before any real compute — and every step receives
it. That file *is* the experiment: it's also archived to MLflow so any
run can be reproduced exactly.

```
   you edit                 platform                you run
┌─────────────┐   push   ┌──────────────┐  ~min  ┌──────────────────────┐
│ config/     │ ───────► │ build+release│ ─────► │ Argo UI submit form  │
│ pipeline.py │          └──────────────┘        │ = your config tree   │
│ steps/*/    │                                  └──────────┬───────────┘
└─────────────┘                                             ▼
                                    [compose] → params.yaml → all steps
```

---

## 2. Anatomy of the config tree

```yaml
# config/train.yaml (excerpt — a SECTION with scalar leaves)
train:
  epochs: 100        # total training epochs
  batch_size: 16     # -1 for auto-batch
  cos_lr: true       # plain booleans are fine
  loss:
    pose: 12.0       # nested scalars work too
```

```yaml
# config/train/optimizer/sgd.yaml (one OPTION of a config GROUP)
name: "SGD"
lr: 0.01
momentum: 0.937
weight_decay: 0.0005
```

```yaml
# config/config.yaml — the defaults list picks each group's default
defaults:
  - _self_
  - train
  - train/optimizer: sgd
  - image_processing: mosaic_default
  # ...
```

What each thing becomes on the submit form:

| you write in config/ | what happens |
|---|---|
| a scalar leaf (`train.epochs: 100`) | a form field named by its path, dots→dashes: **`train-epochs`**, default `100` |
| a nested scalar (`train.loss.pose`) | same rule: **`train-loss-pose`** |
| a **boolean** (`train.amp: true`) | a plain form field (`true`/`false`) — no special treatment |
| a **group directory** (`train/optimizer/*.yaml`) | ONE dropdown (**`train-optimizer`**) whose options are the file names; picking one swaps the whole subtree |
| a **list** or other complex structure | NOT a form field — change it by swapping a group option or via the ADVANCED override |

Some fields are dropdowns with a fixed set of valid values because the
platform declares them (e.g. `data-source`: lakefs/s3,
`quantization-mode`: none/ptq/qat) — submitting anything else fails the
run immediately, with the allowed values listed in the error.

**The boolean non-rule:** booleans are just values now. Write
`amp: true`, submit `false`, your step reads a real `False`. (If you
used the previous Typer-based version of this platform: the
true/false-Enum workaround is gone.)

---

## 3. Add a parameter

Add a leaf to the tree. That's the whole task.

```yaml
# config/train.yaml
train:
  ...
  gradient_clip: 10.0   # NEW
```

Push. The form now has `train-gradient_clip` with default `10.0`, and
every step that declares `reads=["train"]` sees
`cfg["train"]["gradient_clip"]` in its params — no pipeline.py change,
no step-code change (unless you want to act on it), no wiring.

To add a *behavior variant* instead of a single knob, add a new option
file to a group:

```yaml
# config/train/optimizer/lion.yaml  (NEW file = new dropdown option)
name: "Lion"
lr: 0.0003
momentum: 0.95
weight_decay: 0.01
```

Push. The `train-optimizer` dropdown now offers `lion`.

---

## 4. Anatomy of pipeline.py (the whole file)

`pipeline.py` declares structure and nothing else — this is the real,
complete step list from this repo:

```python
from kubecore.authoring import pipeline, step  # the platform's engine provides this

with pipeline("ml-pipeline") as p:  # the platform renames it to {app}-pipeline at release
    validate = step("config-validation", reads=["data", "model", "train"])
    train = step("model-training", gpu=True, needs=[validate], hpc_time_limit="12h",
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

Per step:

- **`reads=[...]`** — which top-level config sections this step
  consumes. This is a declared contract: if a section you read is ever
  renamed or deleted, the *release fails* with a clear error instead of
  your step crashing at 2am.
- **`gpu=True`** — the GPU switch. One flag; node routing, GPU
  reservation, everything else is automatic.
- **`needs=[...]`** — ordering + data: every output of a needed step is
  fed to yours automatically.
- **`outputs=[...]`** — small result files your step writes to
  `/work/output/<name>.json`.
- **`when=...`** — conditional execution keyed on any form parameter
  (parameter name = config path with dashes: `quantization.mode` →
  `quantization-mode`).
- **`disk="20Gi"`** — more scratch disk for the step (a Kubernetes size).
- **`hpc_time_limit="12h"`** — the HPC (Slurm) time limit, GPU steps only (§8.1).
- **`resumable=`**, **`checkpoint_every=`**, **`idle_timeout=`** — optional
  resilience settings, see §8.1 and `docs/resilience.md`.

---

## 5. Add a step

> **New here? Read [ADD-A-STEP.md](ADD-A-STEP.md)** — a complete, copy-paste,
> zero-assumptions walkthrough (every command, every gotcha, how to verify).
> The version below is the quick reference once you know the shape.

Three small edits — say we add `model-evaluation` after training:

1. **config**: add a section for it (if it needs its own knobs):

   ```yaml
   # config/evaluation.yaml
   evaluation:
     split: "val"
     iou_threshold: 0.5
   ```

   and register it in `config/config.yaml`'s defaults list: `- evaluation`.

2. **code**: `mkdir steps/model_evaluation` with a Dockerfile and
   `app/entry.py` that reads its slice (this is the pattern of
   `steps/model_training/app/entry.py`):

   ```python
   import argparse
   import yaml

   READS = ["evaluation"]

   def main() -> None:
       parser = argparse.ArgumentParser()
       parser.add_argument("--params", required=True)
       args, _ = parser.parse_known_args()
       cfg = yaml.safe_load(args.params)
       config = {section: cfg[section] for section in READS}
       ...

   if __name__ == "__main__":
       main()
   ```

3. **pipeline.py**: one line —

   ```python
   evaluate = step("model-evaluation", reads=["evaluation"], needs=[train])
   ```

Push. Done. The form gains `evaluation-split` and
`evaluation-iou_threshold` (plus automatic per-run sizing knobs for the
new step), and it runs after training with `training-result` fed in.

---

## 6. Swap behavior wholesale (groups)

Groups exist so you can swap an entire parameter subtree in one move —
on the form (pick `adamw` in the `train-optimizer` dropdown) or as the
new default in the repo (edit one line of `config/config.yaml`):

```yaml
defaults:
  - train/optimizer: adamw     # was: sgd
```

Everything inside the option file (`lr`, `momentum`, `weight_decay`, …)
changes together — no field-by-field editing, no forgotten stragglers.
Use groups for anything with internally-consistent presets: optimizers,
augmentation recipes, callbacks, QAT settings.

**How a group interacts with its individual fields on the form:**
fields you leave untouched follow the group you selected; fields you
explicitly change win over the group. Picking `adamw` gives you the
full AdamW preset; picking `adamw` *and* setting
`train-optimizer-lr` to `0.005` gives you AdamW with your lr.

---

## 7. Delete a step

1. Remove its `step(...)` line; remove it from any other step's
   `needs=[...]`.
2. Delete `steps/<name>/`; delete (or keep) its config section — if you
   delete the section, also remove it from `config/config.yaml`'s
   defaults list and from any remaining step's `reads=`.

Miss the `reads=` cleanup and the release fails immediately with the
exact section and step named (see §10) — nothing silently breaks.

---

## 8. GPU & resources

- `step("model-training", gpu=True, ...)` is the entire GPU
  configuration.
- Per-run sizing is automatic: every step gets `{step}-cpu` and
  `{step}-mem` fields on the submit form (defaults = a whole node).
  Dial a smoke-run down at submit time; you never configure this.

---

### 8.1 Running steps on HPC (Slurm)

When your project has HPC, every step's `{step}-class` dropdown also offers the HPC classes (named `meluxina-*`,
after today's provider): a GPU class, a CPU class and a large-memory class. Pick one for a step and that step
runs as a Slurm job on HPC; every other step keeps its cluster class. There is no global switch: placement is per
step, per run (training on `meluxina-gpu`, quantization on `gpu-small`, registration on the cluster is a normal
combination).

What happens for an HPC step: the step in the run view is a small *follower* (the template `hpc-follower`). It
asks for a job, prints the job's phase and the queue's reason for waiting, and writes the step's declared outputs
when the job ends. The platform checks the job against your project's HPC allocation, copies the dataset version
to the HPC filesystem (cached by repository and commit), runs the same command the step would run in its image,
and hands the outputs to the next step exactly as if it had run on the cluster. Cancelling or deleting the run
cancels the job. You never name a queue or a quality of service: the platform picks the lane and, from earlier
runs, a time limit (never above yours). A project can be blocked by its own allocation, GPU, CPU, FPGA and large
memory each separately; the reasons you can see (`WaitingForBudget`, `BudgetBlocked`, `SiteLimit`, `SubmitLimit`,
`GPUViolation`, `TooLongNotResumable`) are in `docs/slurm.md`.

**What `pipeline.py` may not do.** Only the platform chooses the account steps run as and the pod settings. A
`pipeline.py` that sets `serviceAccountName` (on the pipeline or a step), sets `podSpecPatch`, or names a step
template `hpc-follower` fails to build with a message naming the step, and nothing is released until you remove it.
Use `step(...)` options and the `{step}-class` / `-cpu` / `-mem` form fields instead.

Give long steps a time limit (default 4 h):

```python
train = step("model-training", gpu=True, needs=[validate], hpc_time_limit="12h",
             reads=["experiment", "data", "model", "train"])
```

`hpc_time_limit` takes minutes or hours (`"90m"`, `"12h"`, `"720"`) and applies to `gpu=True` steps; Slurm ends the
job when it elapses. It is a ceiling: the platform may ask for less and runs once more at your limit if a shorter
one cut the job. Queue time is the one variable you do not control: minutes to hours, depending on the load.

**Longer than the longest HPC run.** A step whose limit is above the site's longest run (for example two days) is
never cut silently. If it is resumable (declared with `resumable=`, or learned: an earlier continuation loaded its
checkpoint and got further) it runs as capped attempts that continue from the checkpoint, up to 5 continuations. If
it is not, it is refused before it is sent, with `TooLongNotResumable` and the two fixes: make it resumable or
shorten it.

Resilience needs nothing in `pipeline.py` or in the steps: every step gets `KAOS_CHECKPOINT_DIR`, a lost step is run
again (at most twice, never for its own failure), and `kaos_lib` hooks Ultralytics so training saves into that folder
and resumes from it (`best.pt` included). The platform keeps the folder safe and restores it. Optional step
settings, all in `docs/resilience.md`:

| setting | meaning |
|---|---|
| `resumable=True`, `"3m"` | declare the step resumable; the string is how long before the HPC time limit it is told to save (default 10 m) |
| `resumable=False` | opt out: never continued from a checkpoint on HPC |
| `checkpoint_every="20m"`, `False` | how often the checkpoint folder is copied; `False` turns copies off |
| `idle_timeout="40m"`, `0` | the idle rule: no GPU use and no new log line for 20 minutes ends the step as `Idle`; `0` turns it off |

---

## 9. Data flow between steps

**Parameters (small)**: declare `outputs=["training-result"]`, write
`/work/output/training-result.json`, and every step that `needs=` yours
receives it automatically (`--training-result <content>`). Keep these
small (KBs) — manifests, metrics, references.

**Your experiment config**: arrives resolved in `--params`; read only
your declared sections.

**THE DATA RULE — steps do not share a filesystem.** Each step runs on
its own machine with its own disk. Move data like this:

| data | mechanism |
|---|---|
| small manifests / results | `outputs=` parameters (above) |
| datasets | lakeFS — each dataset is its own repository `dataset-<name>` (layout at the root); the run's `dataset` = `<repo>@<version>`; the platform pins the commit once per run and gives every step that lock (`KAOS_LIB_LOCK`), so `datasets.resolve` and `datasets.materialize` read it |
| checkpoints | the app's outputs repository `outputs-<app>` on `main` — `cfg["platform"]["checkpoints"]["bucket"]/["prefix"]` |
| metrics, artifacts, models | MLflow (below) |

---

## 10. Local iteration loop (authoring preview only — not a way to run the pipeline)

This is a **render preview**, not a run. It lets you check your DAG and see the
resolved `params.yaml` on a laptop before you push. **Pipelines only run on the
platform, submitted from the run form** (§12) — there is no local execution
and no local submit. The preview needs nothing but Python:

```bash
kaosctl render --context pipeline-context.yaml     # -> out/raw-workflow-template.yaml (your DAG), with the platform's engine
kaosctl dev                                        # one Python environment per step, from its Dockerfile
kaosctl step run model-training --set train.epochs=5   # one step, composed params
```

`kaosctl render` fetches the pipeline engine the platform released for your app, so a render that passes
here passes in the platform's CI. `kaosctl step run` composes the same params the platform composes
and hands them to the step on `--params`, exactly what the step receives at run time (`--set key=value`
is an override as on the run form; `--params-file` uses a params file you already have). Your app holds
no copy of the engine. `--context` is the platform's pipeline-context file of your app; without one,
name an engine release with `--engine-wheel` (see the kaos-lib `render` reference).

What a mistake looks like: a step that reads a config section that does not exist, an override of a key
that is not in the tree, or a value outside a dropdown's options each fail at render or at compose time,
in seconds, with a message that names the key.

The same checks run when you submit: a bad submission stops the run at
the first step, before any GPU time is spent.

---

## 11. Using platform tools from step code

Two ways in, both zero-config:

**The `platform` section of your params** — every resolved params.yaml
contains it (you don't declare or read= it, it's just there):

```python
cfg = yaml.safe_load(args.params)
bucket = cfg["platform"]["checkpoints"]["bucket"]     # "outputs-yolo" (the app's outputs repository)
prefix = cfg["platform"]["checkpoints"]["prefix"]     # "main/checkpoints" (branch/path; add <experiment>/<run>/)
dataset = cfg["dataset"]                               # "dataset-default@latest" — the run's dataset pick
```

**Ambient environment variables** — set on every step container:

| env var | what it is |
|---|---|
| `MLFLOW_TRACKING_URI` | your project's MLflow tracking server |
| `LAKEFS_ENDPOINT` | your project's lakeFS endpoint |
| `LAKEFS_ACCESS_KEY` / `LAKEFS_SECRET_KEY` | lakeFS credentials (managed secret) |

MLflow just works:

```python
import mlflow
mlflow.set_experiment(cfg["experiment"]["name"])
with mlflow.start_run():
    mlflow.log_metric("mAP50", 0.87)
```

Datasets and results go through kaos-lib (`kaos_lib`; see `docs/datasets-in-the-steps.md`):
`datasets.materialize(DatasetRef.parse("<repo>@<ref>"), prefix="folder/")` reads any repository at a pinned ref,
`outputs.run(...)` writes results. You write no endpoint and no credential: the step's environment has
them.

When you run a step locally with `kaosctl step run`, no platform endpoint or secret is available: a step that needs one stops with its own error, or runs on its fallback.

---

## 12. Running the pipeline (from the run form only)

In the console open **Pipelines** → your pipeline → **Run** (or the workflow template in Argo, then **Submit**):

- **`pipeline-info`** (first field) — read-only cheat-sheet; leave it.
- **Group dropdowns** (`model`, `train-optimizer`, `train-callbacks`,
  `train-qat`, `image_processing`) — swap whole presets. Fields you
  leave untouched follow the group you pick; fields you explicitly
  change win over the group.
- **Your scalar fields** — every leaf of your tree
  (`train-epochs`, `dataset`, `train-loss-pose`, …), plus dropdowns the
  platform fills live (`{step}-class` lists the machine classes you may use for each step;
  `dataset` is `<dataset-repo>@<version>`, default
  `dataset-default@latest`, its choices are your datasets and their versions) and
  fixed-choice fields it validates (`data-source`, `quantization-mode`).
- **Per-step sizing** — `{step}-cpu` / `{step}-mem`.
- **`config` (ADVANCED)** — normally empty. When non-empty it is an
  override YAML merged LAST over everything above, for example
  `train: {epochs: 5}`. Keys must exist in the tree — a typo in the YAML fails the run
  immediately (§10). The `platform:` section is off-limits: an
  ADVANCED override that touches it is rejected.

Defaults are runnable as-is: Submit with no changes = a standard full
training run.

---

## 13. What the platform does when you push

You never run the render or the image builds by hand. The platform does it and reports back on the commit; there
are no workflows to add to your repository.

**On a pull request to `dev`** (the `wft-render` check): your pipeline is rendered exactly the way it will be after
the merge, and the check gates the pull request. Green means your DAG builds, every `reads=` section exists, every
step is declared, every step has a buildable `steps/<dir>/Dockerfile`, there are no duplicate form parameters, and
`pipeline.py` sets no service account or pod patch (§8.1). Its link opens the run; if the render fails, the error
is in that run's log. Nothing is released on a pull request.

**On merge to `dev`** (the `kubecore-ml-ci` status): the same render runs and the new pipeline version is
released. Every `steps/*/` whose contents changed is built and wired in as `image-<step>`. The steps to build are
found from the folders, so a step you add is built with no further setup.

So the whole loop for adding a step (§5) is: add `steps/<name>/` + a config section + one `pipeline.py` line →
**open a pull request** → `wft-render` checks it → merge → `kubecore-ml-ci` builds your step image and releases the
pipeline → it is ready to run from the form.

---

## 14. Pitfalls FAQ

**"My new parameter didn't show up on the form."**
Is it inside a *list* or other complex structure? Only scalar leaves
and groups reach the form. Move it to a scalar leaf, or change it via
group options / the ADVANCED override.

**"I renamed a config section and now the release fails."**
That's the `reads=` gate doing its job — the error names the step still
reading the old name. Update that step's `reads=` (and its `READS` in
entry code).

**"Two leaves ended up with the same form name."**
Possible only if your key names contain dashes that collide with the
path mapping (e.g. `train.a-b` vs `train.a.b`). The release fails with
`form-name collision: ...`; rename one key.

**"How do I make something a dropdown?"**
Make it a group: a directory of option files. Scalars are free-text
fields; groups are dropdowns.

**"I picked a group option but my run used the old values."**
It didn't — untouched fields follow the group (§6). If a value really
didn't change, you (or a script) explicitly set that field, and
explicitly-set fields win over the group by design.

**"My run failed with `Invalid value ... expected one of [...]`."**
That field has a fixed set of valid values (e.g. `quantization-mode`:
none/ptq/qat). Pick one of the listed values — the check exists to
stop a typo from wasting a training run.

**"My step needs a value from a section it doesn't read."**
Add the section to its `reads=` (and entry `READS`). Reading
undeclared sections works at runtime but is unprotected — the render
gate only defends what you declare.

**"Booleans?"**
Plain values. `true`/`false` in YAML, on the form, everywhere.

**"Step didn't rebuild after my push."**
A push only rebuilds steps whose `steps/<dir>/` changed. Config-tree or
pipeline.py changes re-release the pipeline without rebuilding images —
that's normal; parameters live in the template, not the images.

**"Can I put big things (file contents, long lists) in config?"**
The resolved params.yaml is capped (200KB, checked with a clear error).
Config is for knobs; bulk data goes through lakeFS/checkpoints (§9).

---

## 15. Running the tests

One command tests your step code (`step_tests/`):

```bash
pip install "$(sed -n 's/^ARG KAOS_LIB_PACKAGE="\(.*\)"$/\1/p' steps/*/Dockerfile | head -1)" \
    pytest boto3 psutil pydantic pydantic-settings mlflow-skinny    # once, in a virtual environment
python -m pytest
```

That installs exactly the requirement in `KAOS_LIB_PACKAGE` in the step Dockerfiles (run it from the repo root).
A step image copies only its `app/` folder. `step_tests/test_repo_layout.py` fails if a `kubecore/` folder or a
`steps/compose_and_validate/` folder comes back: the platform runs its own engine.

The step tests use kaos-lib's in-memory lakeFS (`step_tests/fake_lakefs.py`, vendored at the same pin),
so no network or cluster is needed.
