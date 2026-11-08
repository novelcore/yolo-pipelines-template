"""The checkpoint the training step hands to qat-finetune / model-quantization is never empty.

PRD-HPC-1461 AC-13 (2026-11-08): a training that was continued after exit 85 ended with no best.pt, so
``best_checkpoint_s3`` was "" and qat-finetune died with "'str' object has no attribute 'to'" (``YOLO("")``
yields a model that is the empty string). Training now falls back to last.pt (kaos-lib also restores best.pt
on the continuation), and the consumers refuse an empty or non-YOLO checkpoint with a message that says why.
"""
import types

import pytest
from conftest import import_in

torch = pytest.importorskip("torch", reason="the checkpoint tests need torch (CPU is enough)")
pytest.importorskip("torchao.quantization.pt2e.quantize_pt2e", reason="qat_service imports torchao")
ultralytics = pytest.importorskip("ultralytics", reason="the checkpoint tests load real Ultralytics checkpoints")


@pytest.fixture
def training():
    return import_in("model_training", "app.services.model_training")


@pytest.fixture
def qat():
    return import_in("qat_finetune", "app.services.qat_service")


@pytest.fixture
def quantization():
    return import_in("model_quantization", "app.services.quantization_service")


def make_service(training, uploads):
    service = training.TrainingService(s3_client=None, mlflow_tracking_uri="")
    service._upload_to_s3 = lambda path, bucket, key: uploads.append((path.name, bucket, key))
    return service


def params():
    return types.SimpleNamespace(checkpoint_bucket="outputs-app", checkpoint_prefix="main/checkpoints",
                                 experiment_name="exp")


def weights_dir(tmp_path, *names):
    folder = tmp_path / "run" / "weights"
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_bytes(b"w")
    return tmp_path / "run"


# ------------------------------------------------------------------------------------------- training step

def test_should_return_best_pt_uri_and_upload_both_when_both_exist(training, tmp_path):
    uploads = []
    uri = make_service(training, uploads)._upload_final_weights(params(), weights_dir(tmp_path, "best.pt", "last.pt"))
    assert uri == "s3://outputs-app/main/checkpoints/exp/best.pt"
    assert [u[0] for u in uploads] == ["best.pt", "last.pt"]


def test_should_fall_back_to_last_pt_uri_when_a_continued_run_has_no_best_pt(training, tmp_path):
    uploads = []
    uri = make_service(training, uploads)._upload_final_weights(params(), weights_dir(tmp_path, "last.pt"))
    assert uri == "s3://outputs-app/main/checkpoints/exp/last.pt"
    assert [u[0] for u in uploads] == ["last.pt"]


def test_should_raise_instead_of_returning_an_empty_uri_when_no_weights_exist(training, tmp_path):
    uploads = []
    with pytest.raises(training.TrainingError, match="no weights"):
        make_service(training, uploads)._upload_final_weights(params(), weights_dir(tmp_path))
    assert uploads == []


def test_should_return_the_same_uri_when_called_twice(training, tmp_path):
    service = make_service(training, [])
    run = weights_dir(tmp_path, "last.pt")
    assert service._upload_final_weights(params(), run) == service._upload_final_weights(params(), run)


# ------------------------------------------------------------------------------------------- consumers

@pytest.mark.parametrize("empty", ["", None])
def test_should_refuse_an_empty_checkpoint_path_in_qat(qat, empty):
    with pytest.raises(ValueError, match="best_checkpoint_s3 is empty"):
        qat.QATService("")._resolve_checkpoint(empty, "/tmp")


@pytest.mark.parametrize("empty", ["", None])
def test_should_refuse_an_empty_checkpoint_path_in_quantization(quantization, empty):
    service = quantization.QuantizationService("")
    with pytest.raises(quantization.QuantizationError, match="best_checkpoint_s3 is empty"):
        service._resolve_checkpoint(empty, "/tmp")


def test_should_keep_a_local_checkpoint_path_unchanged(qat):
    assert qat.QATService("")._resolve_checkpoint("/work/runs/x/weights/best.pt", "/tmp") == "/work/runs/x/weights/best.pt"


def tiny_model():
    return ultralytics.YOLO("yolov8n.yaml").model


@pytest.mark.parametrize("shape", ["model", "ema", "model_none_ema_only"])
def test_should_load_every_ultralytics_checkpoint_dict_shape_as_a_headless_module(qat, tmp_path, shape):
    """last.pt/best.pt after strip_optimizer carry the weights under `model` (ema None) or, mid-run, `ema`."""
    model = tiny_model()
    ckpt = {"model": model, "ema": None, "train_args": {"task": "detect"}}
    if shape == "ema":
        ckpt = {"model": None, "ema": model, "train_args": {"task": "detect"}}
    elif shape == "model_none_ema_only":
        ckpt = {"model": model, "ema": model, "train_args": {"task": "detect"}}
    path = tmp_path / "best.pt"
    torch.save(ckpt, path)

    module = qat.QATService("")._load_headless_module(str(path), "cpu")

    assert isinstance(module, torch.nn.Module)
    assert module.model[-1].training is True          # the head-exclusion recipe (CON-03)


def test_should_name_the_cause_when_the_checkpoint_does_not_load_as_a_model(qat):
    with pytest.raises(ValueError, match="did not load as a YOLO model"):
        qat.QATService("")._load_headless_module("", "cpu")
