"""The checkpoint the training step hands to qat-finetune and model-quantization is never empty, and they read it
from the model `checkpoints` that model-training saved (PRD-HPC-1461 AC-13)."""
import os
import types
from pathlib import Path

import pytest
from conftest import composed_settings, import_in, make_dataset
from kaos_lib import testing

# The QAT and quantization steps import torch, torchao and ultralytics when they are loaded.
pytest.importorskip("torch", reason="needs torch (CPU is enough)")
pytest.importorskip("torchao.quantization.pt2e.quantize_pt2e", reason="qat_service imports torchao")
pytest.importorskip("ultralytics", reason="the steps import ultralytics")

TRAIN = {"best_checkpoint": "best.pt", "mlflow_run_id": "mlflow-1"}


def inputs(tmp_path, *extra):
    """The folders the later steps read: the model `checkpoints` and the dataset `prepared`."""
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "best.pt").write_bytes(b"weights")
    folders = {"checkpoints": checkpoints, "prepared": make_dataset(tmp_path / "prepared")}
    for name in extra:
        folders[name] = tmp_path / name
        folders[name].mkdir()
        (folders[name] / "model_int8.tflite").write_bytes(b"int8")
    return folders


def capture(monkeypatch, entry, result):
    seen = {}

    class Manager:
        def run(self, **kwargs):
            seen.update(kwargs)
            (Path(kwargs["export_dir"]) / "model_int8.tflite").write_bytes(b"int8")  # what the service saves
            return types.SimpleNamespace(model_dump=lambda: result)

    monkeypatch.setattr(entry, "Manager", Manager)
    monkeypatch.setattr(entry.tracking, "mlflow", lambda *a, **k: None)
    return seen


# ------------------------------------------------------------------------------------------- the entries

def test_qat_fine_tunes_the_checkpoint_training_saved_on_the_dataset_training_read(tmp_path, monkeypatch):
    entry = import_in("qat_finetune", "app.entry")
    seen = capture(monkeypatch, entry, {"tflite": "model_int8.tflite", "mlflow_run_id": "mlflow-2"})
    with testing.running(tmp_path / "run", kind="train", name="qat-finetune", params=composed_settings(),
                         inputs=inputs(tmp_path), upstream={"model-training": TRAIN},
                         produces=[("model", "qat")]) as run:
        entry.main()
    assert Path(seen["fp32_checkpoint_path"]).read_bytes() == b"weights"
    assert (Path(seen["dataset_dir"]) / "data.yaml").is_file()
    assert seen["source_mlflow_run_id"] == "mlflow-1" and seen["qat_epochs"] == 10
    assert Path(seen["export_dir"]).name == "qat"
    assert run.result["values"] == {"tflite": "model_int8.tflite", "mlflow_run_id": "mlflow-2"}


def test_ptq_works_on_a_copy_of_the_checkpoint_because_the_export_writes_beside_it(tmp_path, monkeypatch):
    entry = import_in("model_quantization", "app.entry")
    seen = capture(monkeypatch, entry, {"tflite": "model_int8.tflite"})
    settings = composed_settings(quantization={**composed_settings()["quantization"], "mode": "ptq"})
    folders = inputs(tmp_path)
    with testing.running(tmp_path / "run", kind="transform", name="model-quantization", params=settings,
                         inputs=folders, upstream={"model-training": TRAIN, "qat-finetune": None},
                         produces=[("model", "quantization")]):
        entry.main()
    checkpoint = Path(seen["fp32_checkpoint_path"])
    assert checkpoint.read_bytes() == b"weights" and checkpoint.parent != folders["checkpoints"]
    assert os.access(checkpoint.parent, os.W_OK), "the export writes beside the checkpoint it is given"
    assert seen["mode"] == "ptq" and seen["tflite_path"] is None


def test_qat_mode_saves_the_int8_model_qat_finetune_made(tmp_path, monkeypatch):
    entry = import_in("model_quantization", "app.entry")
    seen = capture(monkeypatch, entry, {"tflite": "model_int8.tflite"})
    settings = composed_settings(quantization={**composed_settings()["quantization"], "mode": "qat"})
    with testing.running(tmp_path / "run", kind="transform", name="model-quantization", params=settings,
                         inputs=inputs(tmp_path, "qat"),
                         upstream={"model-training": TRAIN,
                                   "qat-finetune": {"tflite": "model_int8.tflite", "mlflow_run_id": "mlflow-2"}},
                         produces=[("model", "quantization")]):
        entry.main()
    assert Path(seen["tflite_path"]).read_bytes() == b"int8" and seen["qat_run_id"] == "mlflow-2"


def test_a_step_that_needs_the_qat_model_but_qat_did_not_run_fails_naming_it(tmp_path, monkeypatch):
    entry = import_in("model_quantization", "app.entry")
    capture(monkeypatch, entry, {})
    settings = composed_settings(quantization={**composed_settings()["quantization"], "mode": "qat"})
    with testing.running(tmp_path / "run", kind="transform", name="model-quantization", params=settings,
                         inputs=inputs(tmp_path), upstream={"model-training": TRAIN, "qat-finetune": None},
                         produces=[("model", "quantization")]):
        with pytest.raises(Exception, match="qat-finetune.*did not run"):
            entry.main()


# ------------------------------------------------------------------------------------------- the services

@pytest.mark.parametrize("empty", ["", None])
def test_should_refuse_an_empty_checkpoint_path_in_qat(empty):
    qat = import_in("qat_finetune", "app.services.qat_service")
    with pytest.raises(ValueError, match="names no checkpoint"):
        qat.QATService()._check_checkpoint(empty)


@pytest.mark.parametrize("empty", ["", None])
def test_should_refuse_an_empty_checkpoint_path_in_quantization(empty):
    quantization = import_in("model_quantization", "app.services.quantization_service")
    with pytest.raises(quantization.QuantizationError, match="names no checkpoint"):
        quantization.QuantizationService()._check_checkpoint(empty)


def test_should_keep_a_local_checkpoint_path_unchanged():
    qat = import_in("qat_finetune", "app.services.qat_service")
    assert qat.QATService()._check_checkpoint("/work/runs/x/weights/best.pt") == "/work/runs/x/weights/best.pt"


def tiny_model():
    import ultralytics
    return ultralytics.YOLO("yolov8n.yaml").model


@pytest.mark.parametrize("shape", ["model", "ema", "model_none_ema_only"])
def test_should_load_every_ultralytics_checkpoint_dict_shape_as_a_headless_module(tmp_path, shape):
    """last.pt/best.pt after strip_optimizer carry the weights under `model` (ema None) or, mid-run, `ema`."""
    import torch

    qat = import_in("qat_finetune", "app.services.qat_service")
    model = tiny_model()
    ckpt = {"model": model, "ema": None, "train_args": {"task": "detect"}}
    if shape == "ema":
        ckpt = {"model": None, "ema": model, "train_args": {"task": "detect"}}
    elif shape == "model_none_ema_only":
        ckpt = {"model": model, "ema": model, "train_args": {"task": "detect"}}
    path = tmp_path / "best.pt"
    torch.save(ckpt, path)

    module = qat.QATService()._load_headless_module(str(path), "cpu")

    assert isinstance(module, torch.nn.Module)
    assert module.model[-1].training is True          # the head-exclusion recipe (CON-03)


def test_should_name_the_cause_when_the_checkpoint_does_not_load_as_a_model():
    qat = import_in("qat_finetune", "app.services.qat_service")
    with pytest.raises(ValueError, match="did not load as a YOLO model"):
        qat.QATService()._load_headless_module("", "cpu")
