# model-training

Trains the YOLO pose model (YOLOv8 / YOLOv11) on the dataset commit that `dataset-loading` pinned.

- **Runs on:** a GPU (`gpu=True` in `pipeline.py`); on Slurm when `model-training-class` is an HPC class (`meluxina-*`).
- **Reads:** the form sections `experiment`, `dataset`, `data`, `model`, `train`, `image_processing`, `logging`. Own datasets stream from the project's lakeFS S3 gateway; shared datasets and Slurm runs are materialized locally first.
- **Writes:** checkpoints (`best.pt`, `last.pt`, `epoch_NNNN.pt`, exports) to `outputs-<pipeline>` under `checkpoints/<experiment>/<run>/`, metrics and a logged model `<experiment>-best` to MLflow. A retried step resumes from its last checkpoint.
- **Output:** `training-result` for the later steps.
- **Settings:** [docs/parameters.md](../../docs/parameters.md) (bounds are the pydantic models in `app/models/training.py`).

```
steps/model_training/
  app/entry.py        reads the composed params and calls the manager
  app/manager.py      orchestrates one run
  app/models/         pydantic models: TrainingParams, TrainingResult
  app/services/       training, the S3 streaming dataset, lakeFS I/O, resource monitor
  Dockerfile          the image (installs kaos-lib at KAOS_LIB_PACKAGE)
```

Tests: `step_tests/` ([DEVELOPER.md](../../DEVELOPER.md) section 15). How data and results move: [docs/datasets-in-the-steps.md](../../docs/datasets-in-the-steps.md).
