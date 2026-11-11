# yolo-pipelines-template

**yolo** is the community pipeline for training YOLO pose models on KAOS: give it labelled images, fill in a form, and it trains, logs to MLflow, optionally makes an INT8 (PTQ or QAT) model, and registers the best one. It is built on the [hera pipeline template](https://github.com/novelcore/hera-pipeline-template) (the scaffold for your own pipelines) and is co-owned with Infinite Orbits.

## Start here

**Read the guide: [novelcore.github.io/yolo-pipelines-template](https://novelcore.github.io/yolo-pipelines-template/)**. You need a KAOS account and a project. The journey, with an action and an expected result at every step:

1. [Prepare and upload a yolo dataset](docs/prepare-dataset.md)
2. [Release a version and pick it](docs/release-a-version.md)
3. [Create a yolo pipeline](docs/create-pipeline.md)
4. [Run it: small, then full, PTQ or QAT](docs/run-small-and-full.md)
5. [Results, weights and models in MLflow](docs/results.md)
6. [Use a dataset shared with you](docs/shared-dataset.md)
7. [Run on HPC (Slurm)](docs/slurm.md)

Something failed? [When something breaks](docs/troubleshooting.md). All the run-form fields: [parameters](docs/parameters.md).

## Quick start

```bash
python3 -m venv ~/.venvs/kaos && . ~/.venvs/kaos/bin/activate
pip install --index-url https://europe-west1-python.pkg.dev/kaos-dev-511509/kaos-lib-rc/simple/ \
    --extra-index-url https://pypi.org/simple/ "kaos-lib[zitadel,mlflow]==0.4.0rc23"
kaosctl login                                    # inside your pipeline's folder
kaosctl dataset create spacecraft --layout yolo --classes spacecraft   # a new dataset has no layout unless you ask
# add images, labels and kpt_shape / flip_idx in data.yaml, then:
kaosctl layout validate ./dataset-spacecraft --layout yolo
kaosctl dataset push ./dataset-spacecraft dataset-spacecraft -m "first images"
kaosctl dataset release dataset-spacecraft       # after merging the pull request
```

Then pick `dataset-spacecraft@v1` in the pipeline's **Run** form. `kaosctl` comes with kaos-lib; its [reference](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/index.html) documents every command.

## The pipeline

```
config-validation → model-training → qat-finetune → model-quantization → model-registration
```

`qat-finetune` and `model-quantization` run only when `quantization-mode` is `qat` or `ptq` (`qat-finetune`: `qat` only). The whole pipeline is `pipeline.py`; **the `config/` tree is the run form**: each plain value is a field and each group folder (`config/model/`, `config/train/optimizer/`, ...) a dropdown.

Long trainings are not lost: a step on a reclaimed machine, a failed node or an HPC (Slurm) time limit is run again
by the platform and training continues from its last checkpoint, with nothing to call. A step that is longer than
the longest HPC run is continued in capped runs when it is resumable, and refused before it is sent otherwise
([Long runs: checkpoints](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/checkpoint.html)).

## For people changing the template

```
pipeline.py            the steps
config/                every setting: a field or a dropdown on the run form
steps/<name>/          one folder per step (README.md in each)
step_tests/            tests of the step code (DEVELOPER.md §15)
scripts/               check-docs.sh, docs_check.py, gen-parameters.py
docs/                  the guide (mkdocs)
```

Your pipeline is `pipeline.py`, `config/` and `steps/`; it holds no copy of the platform's pipeline engine. On your computer, `kaosctl render --context pipeline-context.yaml` renders the pipeline with the same engine the platform uses, `kaosctl dev` sets up each step's Python environment and `kaosctl step run <step>` runs one step ([Check and try a step](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/step-run.html)).

[ADD-A-STEP.md](ADD-A-STEP.md), [DEVELOPER.md](DEVELOPER.md). Check the guide before a pull request: `scripts/check-docs.sh`. `docs/parameters.md` is generated: `python scripts/gen-parameters.py`.

The docs build first runs three checks (`scripts/docs_check.py`), and a finding fails it:

| check | fails on |
|---|---|
| words | a page that names what the platform is built from: the guide is for people who use the platform |
| removed names | a step or a folder that no longer exists |
| links | a link that does not resolve in this guide, in the kaos-lib guide or in the hera guide, and a link from those two into this guide that does not resolve |

The link check downloads the other two repositories; `DOCS_CHECK_OFFLINE=1` checks this one alone and says what it skipped, and `DOCS_CHECK_KAOS_LIB=<folder>` / `DOCS_CHECK_HERA=<folder>` read your own checkouts. One topic has one page: what is true for every pipeline (datasets, checkpoints, checking a step on your computer, kaos-lib versions) is in the [kaos-lib guide](https://storage.googleapis.com/kaos-dev-511509-kaos-lib/docs/index.html), and this guide holds what is specific to yolo. `scripts/docs_check.py` is the same file in kaos-lib and in the hera template: change it in all three.

## License

**GNU Affero General Public License v3.0** (`AGPL-3.0-only`), see [LICENSE](LICENSE). It depends on a training library that is itself AGPL-3.0, and three files adapt that library's internals (`steps/model_training/app/services/s3_dataset.py`, `s3_pose_trainer.py` and `s3_pose_validator.py`), so the template as a whole carries that licence. Releases before this change were published under Apache-2.0, which may be redistributed under AGPL-3.0. A pipeline created from this template inherits these terms for the template's code.
