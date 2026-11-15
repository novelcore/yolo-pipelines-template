# YOLO pose pipeline

This pipeline trains a YOLO pose model (boxes plus keypoints) on your labelled images. You fill in a form and it trains the
model, logs the numbers to MLflow, keeps the weights, can make a smaller INT8 model, and registers the best one.
It is licensed AGPL-3.0 and co-owned with Infinite Orbits.

```mermaid
graph LR
    A["config-validation"] --> P["prepare-data"]
    P --> C["model-training"]
    C --> D["qat-finetune (qat only)"]
    C --> E["model-quantization (ptq or qat)"]
    D --> E
    C --> F["model-registration"]
    E --> F
```

## The files that matter

| path | what it is |
|---|---|
| `pipeline.py` | the six steps and their order |
| `config/` | every setting; each plain value is a field on the run form |
| `steps/<name>/` | one folder for each step: `app/` and a `Dockerfile`; each has its own README |
| `step_tests/` | tests of the steps, see [`step_tests/README.md`](step_tests/README.md) |

What describes this pipeline, in `docs/`; edit it as the pipeline changes:

| file | what it holds |
|---|---|
| [`docs/the-pipeline.md`](docs/the-pipeline.md) | the six steps, what each reads and writes, and what is kept when a step is run again |
| [`docs/parameters.md`](docs/parameters.md) | every field of the run form with its default and bounds. Generated from `config/` by `python scripts/gen-parameters.py`; a test fails when it is out of date |
| [`docs/datasets.md`](docs/datasets.md) | the folder shape the pipeline reads, and the dataset it writes |
| [`docs/slurm.md`](docs/slurm.md) | what to know about the training step on an HPC (Slurm) site |

## Run it on your machine

Install `kaosctl` with pip from your platform's package index. Your platform admin, or the console of your platform, gives you the index and the version to install. To move to a newer version later, run `kaosctl upgrade`. Then run the step tests, which run the steps the way the platform does:

```bash
pip install pytest pillow pydantic psutil "mlflow>=3,<4" && python -m pytest
```

To run one step by hand, use `kaosctl step run` (guide page: try-it-locally).

## The guide

The guide page run-it explains every idea this pipeline uses, one
page at a time. The guide page a-real-pipeline reads this
pipeline's `pipeline.py` with what the first ten pages teach.
