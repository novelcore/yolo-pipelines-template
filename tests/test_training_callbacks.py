"""Every shipped train/callbacks choice must be accepted by the training step.

config/train/callbacks/none.yaml sets patience: 0 (early stopping off) and
TrainingParams rejected it (gt=0), so selecting callbacks=none failed every
training run before it started (kaos e2e ml_run, 2026-10-07).
"""
import importlib.util
import pathlib

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _training_params():
    path = ROOT / "steps/model_training/app/models/training.py"
    spec = importlib.util.spec_from_file_location("training_models", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TrainingParams


def test_every_callbacks_config_is_valid_training_input():
    params = _training_params()
    configs = sorted((ROOT / "config/train/callbacks").glob("*.yaml"))
    assert configs, "no callbacks configs found"
    for cfg in configs:
        patience = (yaml.safe_load(cfg.read_text()) or {}).get("patience", 50)
        params(model_variant="yolov8n-pose.pt", experiment_name="e", dataset_dir="/d",
               output_dir="/o", source="local", patience=int(patience))


def test_negative_patience_is_still_refused():
    params = _training_params()
    try:
        params(model_variant="yolov8n-pose.pt", experiment_name="e", dataset_dir="/d",
               output_dir="/o", source="local", patience=-1)
    except Exception:
        return
    raise AssertionError("patience=-1 was accepted")
