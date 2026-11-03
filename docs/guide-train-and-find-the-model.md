# Guide: train on a version and find the model

**Goal:** a model trained on `dataset-spacecraft@v1`, its numbers in MLflow, and its weights on
your computer.

**You need:** a released version ([Upload images and labels](guide-upload-images.md)). Role
`dev`, `ops` or `admin`.

## 1. Submit

Console → **Pipelines** → your app → **Run** ([Run the pipeline](run-it.md)). Set:

| field | value |
|---|---|
| `experiment-name` | `spacecraft-pose` |
| `dataset` | `dataset-spacecraft@v1` |
| `model` | `yolov8n` (small and fast, for a first run) |
| `train-epochs` | `50` |
| `quantization-mode` | `none` |

**Submit**, and note the run name, for example `my-app-pipeline-x7k2p`.

## 2. Watch it

`config-validation` and `dataset-loading` take seconds. `model-training` takes most of the
time; its log shows the epochs ([Watch it train](watch-it.md)). When `model-registration` is
green, the run is done.

## 3. Find it in MLflow

Console → **Pipelines** → your app → **MLflow** → experiment `spacecraft-pose`.

- The run: **Metrics** (the loss and `val/mAP50_95` curves), **Parameters** (every setting, plus
  `dataset.repo`, `dataset.version` and `dataset.lakefs_commit`), and **Tags**
  (`kaos.outputs_repo`, `kaos.outputs_commit`, `kaos.outputs_path`, `kaos.workflow`).
- The experiment's **Models** tab: `spacecraft-pose-best`, the run's best checkpoint, with its final
  scores.
- **Models** in the top bar: the registered model `object-pose-yolo`, with a new version.

## 4. Get the weights

The weights are in lakeFS, where the run's `kaos.outputs_path` tag points:

```bash
kaosctl dataset pull outputs-<app>@main ./weights --prefix checkpoints/spacecraft-pose/my-app-pipeline-x7k2p/
ls ./weights        # best.pt  last.pt  epoch_0010.pt ...
```

The registered version also has a lakeFS tag, `model-object-pose-yolo-v<N>`, on the same commit.

## 5. Replay its data later

`model-training` records the dataset commit in the run's data lock. To evaluate or retrain on
exactly that data:

```bash
export MLFLOW_TRACKING_URI=https://mlflow-<project>.<project domain>
kaosctl run --lock <mlflow-run-id> -- python my_eval.py ./weights/best.pt
```

**Next:** train a second time with other settings and compare: [A day with yolo](worked-day.md).
