"""model-quantization calibrates on the images of the dataset `prepared` it was given, wherever that dataset's own
data.yaml says its images are (PRD kaos#1911)."""
import contextlib
import types
from pathlib import Path

import pytest
import yaml
from conftest import import_in, make_dataset

# The quantization service imports torch and ultralytics when it is loaded.
pytest.importorskip("torch", reason="needs torch (CPU is enough)")
pytest.importorskip("ultralytics", reason="the step imports ultralytics")


@pytest.fixture
def service():
    return import_in("model_quantization", "app.services.quantization_service").QuantizationService()


def prepared_pointing_elsewhere(tmp_path):
    """A prepared dataset whose data.yaml names the folder of the raw dataset it was made from, which this step
    does not have: the case of a dataset whose `path` is where its author made it."""
    prepared = make_dataset(tmp_path / "prepared")
    (prepared / "data.yaml").write_text(yaml.safe_dump({
        "path": str(tmp_path / "raw-source-not-here"), "train": "images/train", "val": "images/val",
        "test": "images/test", "nc": 1, "names": {0: "ship"}}))
    return prepared


def test_ptq_calibrates_on_the_prepared_dataset_not_on_the_path_in_its_data_yaml(service, tmp_path, monkeypatch):
    """The export collects its INT8 calibration images the way Ultralytics does: check_det_dataset(data=...)."""
    from ultralytics.data.utils import check_det_dataset

    module = types.SimpleNamespace(**type(service)._run_ptq.__globals__)  # the module the service was imported from
    prepared = prepared_pointing_elsewhere(tmp_path)
    seen = {}

    class FakeYolo:
        def __init__(self, checkpoint):
            pass

        def export(self, **kwargs):
            seen["val"] = check_det_dataset(kwargs["data"], autodownload=False)["val"]
            tflite = tmp_path / "model_int8.tflite"
            tflite.write_bytes(b"int8")
            return str(tflite)

    monkeypatch.setitem(type(service)._run_ptq.__globals__, "YOLO", FakeYolo)
    monkeypatch.setattr(module.mlflow, "start_run", lambda **kw: contextlib.nullcontext(
        types.SimpleNamespace(info=types.SimpleNamespace(run_id="r"))))
    for name in ("_save_tflite", "_log_tflite_artifact", "_run_map_delta_and_log", "_log_ptq_run"):
        monkeypatch.setattr(service, name, lambda *a, **k: "model_int8.tflite")
    monkeypatch.setattr(service, "_check_checkpoint", lambda path: path)
    monkeypatch.setattr(service, "_run_parity_and_log",
                        lambda **k: types.SimpleNamespace(parity_passed=True, max_abs_error=0.0))
    params = module.QuantizationParams(
        mode="ptq", fp32_checkpoint_path="best.pt", source_mlflow_run_id="m", dataset_dir=str(prepared),
        output_dir=str(tmp_path / "work"), export_dir=str(tmp_path / "out"), experiment_name="e")
    service._run_ptq(params)
    assert Path(seen["val"]).resolve() == (prepared / "images" / "val").resolve()
    assert len(list(Path(seen["val"]).iterdir())) == 3


def test_the_datasets_own_data_yaml_is_not_changed(service, tmp_path):
    prepared = prepared_pointing_elsewhere(tmp_path)
    before = (prepared / "data.yaml").read_text()
    data_yaml = service._find_data_yaml(str(prepared), str(tmp_path / "work"))
    assert (prepared / "data.yaml").read_text() == before, "the prepared dataset may be a link into a shared cache"
    assert Path(data_yaml).parent == tmp_path / "work"


def test_a_dataset_without_a_data_yaml_is_refused_naming_the_folder(service, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(Exception, match="No YOLO data YAML found"):
        service._find_data_yaml(str(empty), str(tmp_path / "work"))
