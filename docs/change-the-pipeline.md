# Change the pipeline: a setting, a step

This page is for the people who change the template: add a setting, swap what a setting does, or add or remove a step. To
run the pipeline you only need the [journey](index.md).

Your repository is `pipeline.py` (the steps and their order), `config/` (every setting) and `steps/<name>/` (the code of
each step). The platform runs its own pipeline engine: you never copy one into the repository.

## Settings: the config tree is the run form

Every plain value in `config/` is a field of the run form, named by its path with dots turned into dashes. Every folder of
options is a dropdown.

| you write in `config/` | what the run form gets |
|---|---|
| `train.epochs: 100` in `config/train.yaml` | a field `train-epochs`, default `100` |
| `train.loss.pose: 12.0` | a field `train-loss-pose` |
| `train.amp: true` | a field with `true` or `false` |
| the files of `config/train/optimizer/` | one dropdown, `train-optimizer`; each file name is an option |
| a list | no field: change it by swapping an option |

The defaults list in `config/config.yaml` picks each dropdown's default. Fields you leave alone follow the option you pick;
a field you change wins over the option.

**Add a setting:** add a line to a file of `config/`. The field appears on the next release; nothing else is wired. A step
reads it from `ctx.params`, but only from a section that its line in `pipeline.py` names in `reads=`.

**Add an option:** add a file to a group folder, for example `config/train/optimizer/lion.yaml`. To make it the default,
change `- train/optimizer: sgd` to `lion` in `config/config.yaml`.

Some fields accept only certain values (`quantization-mode` is `none`, `ptq` or `qat`): a run with another value is refused
at once and the message lists the allowed ones. A dataset is a setting too: `dataset` holds `dataset-<name>@<version>`, and the
version is chosen on the run form, never in code ([Datasets](datasets.md)).

## What `pipeline.py` says about each step

```python
train = step("model-training", kind="train", gpu=True, needs=[prepare], walltime="12h",
             reads=["experiment", "model", "train", "image_processing", "logging"],
             produces=[model("checkpoints")])
```

| keyword | meaning |
|---|---|
| `kind=` | what the step does: `ingest`, `transform`, `train`, `evaluate`, `publish` or `control`. Required |
| `reads=` | the sections of `config/` the step gets in `ctx.params`, and no others |
| `inputs=` | settings that hold a dataset (`dataset: dataset-x@v1`); the platform pins each to an exact version when the run starts |
| `needs=` | the steps that must finish first; their results and what they wrote are the step's to read |
| `produces=` | what the step writes: `model("name")`, `artifact("name")`, `dataset("name")` |
| `gpu=True` | the step needs a GPU |
| `walltime="12h"` | the longest the step may run on an HPC (Slurm) class |
| `when=` | run the step only when a condition on a field of the run form holds |

Every call and rule is in the kaos-lib guide: [How a step works](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/steps.html)
and [the step API](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/step.html).

## Add a step

1. **Make the folder** `steps/model_evaluation/` with `app/__init__.py` and `app/entry.py`. The step's name in `pipeline.py` is
   `model-evaluation`; the folder uses an underscore.
2. **Write the code.** A step takes no arguments and reads no variable: it starts, reads its inputs, writes its outputs and
   ends with its result.

    ```python
    """model-evaluation: score the trained model on the test split and save a report."""
    from kaos_lib import step


    def main() -> None:
        ctx = step.start()
        weights = step.input("checkpoints") / "best.pt"      # the model model-training saved in this run
        data = step.input("prepared")                        # the dataset version prepare-data wrote in this run
        score = evaluate(weights, data, ctx.params["evaluation"])
        (step.output("report") / "score.json").write_text(f'{{"score": {score}}}')
        step.result(score=score)


    if __name__ == "__main__":
        main()
    ```

3. **Write the `Dockerfile`** next to `app/`: copy the one of `steps/prepare_data/` and add what your code imports with a
   `RUN pip install` line. Keep `ARG KAOS_LIB_PACKAGE`: the platform passes the kaos-lib version.
4. **Add its settings** to `config/` if it has any (`config/evaluation.yaml`, and `- evaluation` in the defaults list).
5. **Declare it** in `pipeline.py`:

    ```python
    evaluate = step("model-evaluation", kind="evaluate", needs=[prepare, train], reads=["evaluation"],
                    produces=[artifact("report")])
    ```

    A step that reads `step.input("prepared")` needs `prepare` in `needs=`; one that reads `step.input("checkpoints")` needs
    `train`. Import `artifact` next to `model` and `dataset` at the top of the file.
6. **Check it on your computer** ([Check your pipeline](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/lint.html),
   [Run a step locally](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/step-run.html)):

    ```bash
    kaosctl lint
    kaosctl step run prepare-data --input dataset=./a-few-files
    kaosctl step run model-evaluation
    ```

7. **Commit and push.** The platform builds the images of the steps whose folder changed and releases the pipeline. A change
   to `config/` or `pipeline.py` alone releases it without rebuilding images.

## Remove a step

Delete its `step(...)` line and remove it from the `needs=` of other steps, then delete `steps/<name>/`. If its settings are
not read by another step, delete the section from `config/` and its line from the defaults list in `config/config.yaml`. A
section that a step still names in `reads=` but that is gone stops the release, naming the step and the section.

## The tests of the steps

`step_tests/` tests the step code with kaos-lib's own test helper, `kaos_lib.testing.running`, which prepares what the
platform hands a step and lets a test call the step's `main()` ([Test a step in your own tests](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/step.html#test-a-step-in-your-own-tests)).
See `step_tests/README.md` for the install line and for the real one-epoch run on the CPU.

## Common questions

**My new setting is not on the form.** Is it inside a list? Only plain values and group folders reach the form.

**A release fails after I renamed a section.** A step still names the old name in `reads=`: the message says which.

**Two settings got the same field name.** Only when a key contains a dash that collides with the path (`train.a-b` and
`train.a.b`); rename one.

**A step did not rebuild after my push.** A push rebuilds only the steps whose folder changed.

**Can I put big values in `config/`?** No: settings are for knobs. Data goes in a dataset, files in a model or an artifact a step
writes ([Results](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/results.html)).
