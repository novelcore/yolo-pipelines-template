# Guide: first dataset to a trained model

**Goal:** from a new yolo app to a trained model and its files, as quickly as possible. Each step
links to the page with the details.

**You need:** a yolo app cloned on your computer, and `kaosctl` signed in inside its folder
([A day with yolo](worked-day.md), morning).

!!! note "Not on `dataset-default`"
    A new project's `dataset-default` is empty. The yolo pipeline needs a `data.yaml`, so a run on
    `dataset-default@latest` (the form's default) stops at `config-validation` with
    "has no data.yaml at the root of its repository". Upload your data first.

## 1. Your dataset

```bash
kaosctl dataset create spacecraft --layout yolo --classes spacecraft
kaosctl dataset pull dataset-spacecraft@latest ./dataset-spacecraft
# add kpt_shape and flip_idx to data.yaml; copy in images/ and one labels/ .txt per image
kaosctl dataset validate ./dataset-spacecraft
kaosctl dataset push ./dataset-spacecraft dataset-spacecraft -m "first 200 images"
```

Merge the pull request in lakeFS, then release `v1`:

```bash
kaosctl dataset release dataset-spacecraft
```

Details: [Upload images and labels](guide-upload-images.md).

## 2. Train

Console → **Pipelines** → your app → **Run**. Set `dataset` = `dataset-spacecraft@v1`,
`experiment-name` = `my-first-run`, `model` = `yolov8n` and `train-epochs` = `10`, then **Submit**.
If `v1` is not in the dropdown yet, wait a few minutes and reopen the form
([Run the pipeline](run-it.md)).

To train on MeluXina, set `model-training-class` to `meluxina-gpu` on the same form
([Where it trains](where-it-trains.md)).

## 3. Your results

- **Numbers, charts and the registered model** are in MLflow ([Read your results](results.md)).
- **Weights** are in `outputs-<app>` in lakeFS, one commit per step per run:

    | folder | what |
    |---|---|
    | `checkpoints/<experiment>/<run>/` | `best.pt`, `last.pt`, `epoch_NNNN.pt`, exports |
    | `quantization/<experiment>/<run>/` | the INT8 `.tflite` (when quantization is on) |

```bash
kaosctl dataset pull outputs-<app>@main ./results --prefix checkpoints/my-first-run/<run>/
```

The commit's details name the dataset, the commit it trained on, and the MLflow run id. If the
training step is retried (for example, its machine was reclaimed), it resumes from its last saved
checkpoint.

**Next:** [Train on a version and find the model](guide-train-and-find-the-model.md).
