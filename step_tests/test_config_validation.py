"""config-validation checks quantization and pretrained weights: not the dataset, not MLflow."""
import pytest
from conftest import ROOT, import_in
from kaos_lib import testing
from kaos_lib.lakefs import LakeFS


def _service():
    mod = import_in("config_validation", "app.services.config_validation")
    return mod, mod.ConfigValidationService()


class _Lake:
    """list_objects over a fixed set of (repository, ref, path)."""

    def __init__(self, *objects):
        self.objects = set(objects)
        self.asked = []

    def list_objects(self, repo, ref, prefix=""):
        self.asked.append((repo, ref, prefix))
        return iter([{"path": path} for r, f, path in self.objects if (r, f) == (repo, ref) and path.startswith(prefix)])


def _lake(monkeypatch, lake):
    monkeypatch.setattr(LakeFS, "connect", classmethod(lambda cls, *a, **k: lake))
    return lake


def test_existing_pretrained_weights_pass(monkeypatch):
    _, svc = _service()
    lake = _lake(monkeypatch, _Lake(("outputs-yolo", "main", "checkpoints/exp/run/best.pt")))
    svc.run({"model": {"pretrained_weights": "outputs-yolo@main/checkpoints/exp/run/best.pt"}}, "yolo", "yolo")
    assert lake.asked == [("outputs-yolo", "main", "checkpoints/exp/run/best.pt")]


def test_missing_pretrained_weights_fail(monkeypatch):
    mod, svc = _service()
    _lake(monkeypatch, _Lake())
    with pytest.raises(mod.ConfigValidationError, match="pretrained weights not found: repo@main/w.pt"):
        svc.run({"model": {"pretrained_weights": "repo@main/w.pt"}}, "yolo", "yolo")


@pytest.mark.parametrize("value", [
    "/somewhere/w.pt",                          # no repository
    "outputs-yolo/main/w.pt",                   # missing @
    "outputs-yolo@main",                        # missing path
    "outputs-yolo@main/",                       # empty path
    "s3://outputs-yolo/main/checkpoints/w.pt",  # the old form
])
def test_pretrained_weights_must_be_empty_or_a_reference(value):
    mod, svc = _service()
    with pytest.raises(mod.ConfigValidationError, match=r"<repository>@<version>/<path>, for example outputs-myapp@main/"):
        svc.run({"model": {"pretrained_weights": value}}, "yolo", "yolo")


def test_nothing_to_look_up_needs_no_lake(monkeypatch):
    _, svc = _service()
    _lake(monkeypatch, None)  # connecting would fail: nothing asks
    svc.run({"model": {"pretrained_weights": None}}, "yolo", "yolo")
    svc.run({"model": {"pretrained_weights": ""}}, "yolo", "yolo")


def test_it_checks_neither_the_dataset_nor_mlflow():
    _, svc = _service()
    svc.run({"dataset": "not a dataset", "model": {}}, "yolo", "yolo")  # no error, no lookup
    text = (ROOT / "steps/config_validation/app/services/config_validation.py").read_text().lower()
    assert "mlflow" not in text and "environ" not in text


def test_the_step_checks_the_settings_it_reads_and_hands_on_that_it_did(tmp_path):
    entry = import_in("config_validation", "app.entry")
    with testing.running(tmp_path, kind="control", params={"model": {}, "quantization": {"mode": "ptq", "calibration_frames": 512}}) as run:
        entry.main()
    assert run.result["values"] == {"checked": True}


def test_a_setting_out_of_bounds_ends_the_step_without_a_result(tmp_path):
    entry = import_in("config_validation", "app.entry")
    with testing.running(tmp_path, kind="control", params={"model": {}, "quantization": {"mode": "ptq", "calibration_frames": 3}}) as run:
        with pytest.raises(Exception, match="calibration_frames"):
            entry.main()
    assert run.result is None
