"""Bad form values must fail in config-validation, before any GPU time is spent.

config-validation has no pydantic, so it mirrors the bounds of the models the later steps use;
these tests pin the mirror to those models. They also pin the batch-size wording of the form.
"""
import importlib.util
from pathlib import Path

import pytest
from conftest import ROOT, import_in


def _model(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _service():
    mod = import_in("config_validation", "app.services.config_validation")
    return mod, mod.ConfigValidationService()


def _bounds(model, field):
    out = {}
    for m in model.model_fields[field].metadata:
        for key in ("ge", "le"):
            if getattr(m, key, None) is not None:
                out[key] = getattr(m, key)
    return out


def test_mirrored_bounds_equal_the_models():
    mod, _ = _service()
    quant = _model("steps/model_quantization/app/models/quantization.py", "m_quant").QuantizationParams
    qat = _model("steps/qat_finetune/app/models/quantization.py", "m_qat").QATParams
    want = {"ge": mod.CALIBRATION_FRAMES_MIN, "le": mod.CALIBRATION_FRAMES_MAX}
    assert _bounds(quant, "calibration_frames") == want
    assert _bounds(qat, "calibration_frames") == want


@pytest.mark.parametrize("mode", ["ptq", "qat"])
@pytest.mark.parametrize("frames", [99, 0, -5, 10001, "50", "abc", ""])
def test_a_bad_calibration_frames_fails_before_training(mode, frames):
    mod, svc = _service()
    with pytest.raises(mod.ConfigValidationError, match="calibration_frames"):
        svc.run({"dataset": "dataset-cats@v1", "quantization": {"mode": mode, "calibration_frames": frames}})


@pytest.mark.parametrize("frames", [100, 512, 10000, "512"])
def test_good_calibration_frames_pass(frames):
    _, svc = _service()
    svc.run({"dataset": "dataset-cats@v1", "quantization": {"mode": "ptq", "calibration_frames": frames}})


def test_the_value_is_ignored_when_no_int8_step_runs():
    _, svc = _service()
    svc.run({"dataset": "dataset-cats@v1", "quantization": {"mode": "none", "calibration_frames": 5}})


def test_the_default_form_values_pass():
    _, svc = _service()
    svc.run({"dataset": "dataset-cats@v1", "quantization": {"mode": "qat", "calibration_frames": 512}})


def test_training_rejects_auto_batch_and_the_form_does_not_promise_it():
    training = _model("steps/model_training/app/models/training.py", "m_train").TrainingParams
    base = dict(model_variant="yolov8n-pose.pt", experiment_name="e", dataset_dir="d", output_dir="o", source="local")
    with pytest.raises(Exception, match="batch_size"):
        training(batch_size=-1, **base)
    text = (ROOT / "config" / "train.yaml").read_text()
    batch_line = next(line for line in text.splitlines() if "batch_size" in line)
    assert "auto" not in batch_line.lower()
