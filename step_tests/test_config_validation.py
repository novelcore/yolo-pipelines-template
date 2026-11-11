"""config-validation checks quantization, pretrained weights and checkpoint resume: not the dataset, not MLflow."""
import pytest
from conftest import ROOT, import_in


def _service():
    mod = import_in("config_validation", "app.services.config_validation")
    return mod, mod.ConfigValidationService()


class _S3:
    """head_object over a fixed key set; list_objects_v2 over the same keys."""

    def __init__(self, keys=()):
        self.keys = set(keys)
        self.heads = []

    def head_object(self, Bucket, Key):
        self.heads.append(f"s3://{Bucket}/{Key}")
        if f"s3://{Bucket}/{Key}" not in self.keys:
            raise KeyError(Key)
        return {}

    def list_objects_v2(self, Bucket, Prefix, MaxKeys):
        return {"Contents": [{"Key": k.split("/", 3)[3]} for k in self.keys if k.startswith(f"s3://{Bucket}/{Prefix}")]}


def _with_store(monkeypatch, svc, s3):
    monkeypatch.setenv("LAKEFS_ENDPOINT", "http://lakefs")
    monkeypatch.setattr(svc, "_s3_client", lambda source: s3)


def test_resume_from_is_read_from_train_checkpointing(monkeypatch):
    """The config key is train.checkpointing.resume_from (config/train.yaml), not data.resume_from."""
    assert "resume_from" in (ROOT / "config" / "train.yaml").read_text()
    assert "resume_from" not in (ROOT / "config" / "data.yaml").read_text()
    mod, svc = _service()
    s3 = _S3({"s3://repo/main/last.pt"})
    _with_store(monkeypatch, svc, s3)
    svc.run({"train": {"checkpointing": {"resume_from": "s3://repo/main/last.pt"}}})
    assert s3.heads == ["s3://repo/main/last.pt"], "the checkpoint was looked up"


def test_a_missing_checkpoint_to_resume_from_fails(monkeypatch):
    mod, svc = _service()
    _with_store(monkeypatch, svc, _S3())
    with pytest.raises(mod.ConfigValidationError, match="checkpoint not found: s3://repo/main/gone.pt"):
        svc.run({"train": {"checkpointing": {"resume_from": "s3://repo/main/gone.pt"}}})


def test_auto_resume_needs_a_checkpoint_under_the_platform_prefix(monkeypatch):
    mod, svc = _service()
    platform = {"checkpoints": {"bucket": "outputs-yolo", "prefix": "main/checkpoints"}}
    cfg = {"train": {"checkpointing": {"resume_from": "auto"}}, "platform": platform}
    _with_store(monkeypatch, svc, _S3())
    with pytest.raises(mod.ConfigValidationError, match="no .pt checkpoint under s3://outputs-yolo/main/checkpoints/"):
        svc.run(cfg)
    _with_store(monkeypatch, svc, _S3({"s3://outputs-yolo/main/checkpoints/exp/best.pt"}))
    svc.run(cfg)


def test_a_resume_in_the_old_data_section_is_not_read(monkeypatch):
    mod, svc = _service()
    s3 = _S3()
    _with_store(monkeypatch, svc, s3)
    svc.run({"data": {"resume_from": "s3://repo/main/gone.pt"}})
    assert s3.heads == []


def test_a_missing_pretrained_weights_file_fails(monkeypatch):
    mod, svc = _service()
    _with_store(monkeypatch, svc, _S3())
    with pytest.raises(mod.ConfigValidationError, match="pretrained weights not found"):
        svc.run({"model": {"pretrained_weights": "s3://repo/main/w.pt"}})


def test_it_checks_neither_the_dataset_nor_mlflow(monkeypatch):
    mod, svc = _service()
    s3 = _S3()
    _with_store(monkeypatch, svc, s3)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://unreachable.invalid")
    svc.run({"dataset": "not a dataset", "model": {}, "data": {"source": "lakefs"}})  # no error, no lookup
    assert s3.heads == []
    for gone in ("_check_dataset", "_check_dataset_path", "_dataset_uri", "_check_mlflow"):
        assert not hasattr(svc, gone), gone
    assert "mlflow" not in (ROOT / "steps/config_validation/app/services/config_validation.py").read_text().lower()


def test_a_check_that_must_run_without_credentials_is_an_error_on_the_platform(monkeypatch):
    mod, svc = _service()
    monkeypatch.setenv("LAKEFS_ACCESS_KEY", "k")   # the platform injected part of the store, not the endpoint
    with pytest.raises(mod.ConfigValidationError, match="LAKEFS_ENDPOINT is not set"):
        svc.run({"model": {"pretrained_weights": "s3://repo/main/w.pt"}})
    with pytest.raises(mod.ConfigValidationError, match="LAKEFS_ENDPOINT is not set"):
        svc.run({"train": {"checkpointing": {"resume_from": "auto"}}})


def test_no_platform_environment_at_all_skips_and_says_so(caplog):
    mod, svc = _service()
    with caplog.at_level("INFO"):
        svc.run({"model": {"pretrained_weights": "s3://repo/main/w.pt"}})
    assert "object-store checks skipped: no platform environment (running locally)" in caplog.text


def test_nothing_to_look_up_needs_no_credentials(monkeypatch):
    mod, svc = _service()
    monkeypatch.setenv("LAKEFS_ACCESS_KEY", "k")
    svc.run({"model": {"pretrained_weights": None}, "train": {"checkpointing": {"resume_from": ""}}})
