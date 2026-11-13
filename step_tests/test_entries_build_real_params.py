"""Every step entry builds the REAL parameters of its manager from what the step API gives it.

`step.input(...)` and `step.output(...)` give a `Path`; the managers' parameter models take text. A test that replaces
the manager (as test_checkpoint_handover.py and test_model_training.py do) cannot see a `Path` where text is expected,
and the quantization step once died on exactly that. Here the real entry calls the real `Manager.run` and builds the
real pydantic parameters; only the service behind the manager (torch, ultralytics, MLflow) is replaced by a recorder.
"""
import contextlib
import importlib
import sys
import types
from pathlib import Path

import pytest
from conftest import composed_settings, make_dataset, step_app
from kaos_lib import testing

TRAIN = {"best_checkpoint": "best.pt", "mlflow_run_id": "mlflow-1"}


def stub_service(module, cls, seen):
    """A module `app.services.<module>` whose `<cls>.run` records the real params and saves an INT8 file."""
    class Service:
        def run(self, params=None, **kwargs):
            seen.update(params.model_dump())
            where = Path(getattr(params, "export_dir", None) or params.checkpoints_dir)
            (where / "model_int8.tflite").write_bytes(b"int8")
            (where / "best.pt").write_bytes(b"w")
            return types.SimpleNamespace(model_dump=lambda: {"tflite": "model_int8.tflite", "mlflow_run_id": "mlflow-2",
                                                             "best_checkpoint": "best.pt"},
                                         mlflow_run_id="mlflow-2", final_map50=0.0, final_map50_95=0.0,
                                         best_checkpoint="best.pt", exported_models={})

    fake = types.ModuleType(f"app.services.{module}")
    setattr(fake, cls, Service)
    sys.modules[f"app.services.{module}"] = fake


def inputs(tmp_path, *extra):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "best.pt").write_bytes(b"weights")
    folders = {"checkpoints": checkpoints, "prepared": make_dataset(tmp_path / "prepared")}
    for name in extra:
        folders[name] = tmp_path / name
        folders[name].mkdir()
        (folders[name] / "model_int8.tflite").write_bytes(b"int8")
    return folders


@contextlib.contextmanager
def entry_of(step_name, module, cls):
    seen = {}
    with step_app(step_name):
        stub_service(module, cls, seen)
        entry = importlib.import_module("app.entry")
        entry.tracking.mlflow = lambda *a, **k: None   # signing in is kaos-lib's
        yield entry, seen


@pytest.mark.parametrize("mode", ["ptq", "qat"])
def test_model_quantization_builds_its_real_params(tmp_path, mode):
    settings = composed_settings(quantization={**composed_settings()["quantization"], "mode": mode})
    extra = ("qat",) if mode == "qat" else ()
    upstream = {"model-training": TRAIN, "qat-finetune": {"tflite": "model_int8.tflite", "mlflow_run_id": "q"} if mode == "qat" else None}
    with entry_of("model_quantization", "quantization_service", "QuantizationService") as (entry, seen), \
            testing.running(tmp_path / "run", kind="transform", name="model-quantization", params=settings,
                            inputs=inputs(tmp_path, *extra), upstream=upstream, produces=[("model", "quantization")]):
        entry.main()
    assert seen["mode"] == mode and isinstance(seen["fp32_checkpoint_path"], str)
    assert Path(seen["fp32_checkpoint_path"]).read_bytes() == b"weights"


def test_qat_finetune_builds_its_real_params(tmp_path):
    with entry_of("qat_finetune", "qat_service", "QATService") as (entry, seen), \
            testing.running(tmp_path / "run", kind="train", name="qat-finetune", params=composed_settings(),
                            inputs=inputs(tmp_path), upstream={"model-training": TRAIN}, produces=[("model", "qat")]):
        entry.main()
    assert isinstance(seen["fp32_checkpoint_path"], str) and Path(seen["fp32_checkpoint_path"]).read_bytes() == b"weights"


def test_model_training_builds_its_real_params_with_pretrained_weights(tmp_path):
    weights = tmp_path / "mine.pt"
    weights.write_bytes(b"w")
    settings = composed_settings()
    settings["model"] = {**settings["model"], "pretrained_weights": str(weights)}
    with entry_of("model_training", "model_training", "TrainingService") as (entry, seen), \
            testing.running(tmp_path / "run", kind="train", name="model-training", params=settings,
                            inputs={"prepared": make_dataset(tmp_path / "prepared")}, produces=[("model", "checkpoints")]):
        entry.main()
    assert seen["pretrained_weights"] == str(weights) and isinstance(seen["dataset_dir"], str)
