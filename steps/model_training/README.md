# model-training

Trains the YOLO pose model (YOLOv8 / YOLOv11) on the dataset version `prepare-data` wrote in the same run.

- **Runs on:** a GPU (`gpu=True` in `pipeline.py`); on HPC (Slurm) when `model-training-class` is an HPC class.
- **Reads:** the settings `experiment`, `model`, `train`, `image_processing`, `logging`, and `step.input("prepared")`: the dataset version `prepare-data` wrote, as files on the machine.
- **Writes:** the model `checkpoints` (`best.pt`, `last.pt`, exports), saved in `outputs-<pipeline>` under `checkpoints/<experiment>/<run>/`; metrics to MLflow. A step run again after a lost machine or an HPC time limit continues from its last checkpoint (nothing to call; see docs/the-pipeline.md), and `best.pt` survives the continuation. The result always names a checkpoint: `best.pt`, or `last.pt` when a continued run wrote no `best.pt`; a run with no weights at all fails.
- **Hands on:** the MLflow run, the checkpoint's file name, the epochs completed, the final validation metrics and the exports.
- **Settings:** [docs/parameters.md](../../docs/parameters.md) (bounds are the pydantic models in `app/models/training.py`).

```
steps/model_training/
  app/entry.py        reads the settings and the dataset and calls the manager
  app/manager.py      orchestrates one run
  app/models/         pydantic models: TrainingParams, TrainingResult
  app/services/       the training service
  Dockerfile          the image (installs kaos-lib at KAOS_LIB_PACKAGE)
```

Tests: `step_tests/` (`step_tests/README.md`), including a real one-epoch run on the CPU. How data and results move: [docs/datasets-in-the-steps.md](../../docs/datasets-in-the-steps.md).
