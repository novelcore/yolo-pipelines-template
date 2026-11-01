"""PRD-1016 F-04/F-05: the model-training step off-cluster (MeluXina) —
local-dir dataset from the staged ref, and checkpoint/weights I/O through the
lakeFS objects API with the run's bearer token instead of the S3 gateway."""
import io
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "steps/model_training"))
from app.services import lakefs_objects  # noqa: E402
import app.entry as entry  # noqa: E402


class _Recorder:
    def __init__(self, body=b"weights"):
        self.calls, self.body = [], body

    def __call__(self, req, timeout=0):
        self.calls.append((req.get_method(), req.full_url, req.headers.get("Authorization"), req.data))
        return io.BytesIO(self.body)


def test_split_key_first_segment_is_the_branch():
    assert lakefs_objects.split_key("main/checkpoints/exp/best.pt") == ("main", "checkpoints/exp/best.pt")
    with pytest.raises(ValueError):
        lakefs_objects.split_key("no-branch")


def test_bearer_mode_needs_both_endpoint_and_token(monkeypatch):
    monkeypatch.delenv("LAKEFS_BEARER_TOKEN", raising=False)
    monkeypatch.setenv("LAKEFS_ENDPOINT", "https://lakefs.example")
    assert not lakefs_objects.bearer_available()
    monkeypatch.setenv("LAKEFS_BEARER_TOKEN", "jwt")
    assert lakefs_objects.bearer_available()


def test_upload_posts_to_branch_objects_api(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKEFS_ENDPOINT", "https://lakefs.example/")
    monkeypatch.setenv("LAKEFS_BEARER_TOKEN", "jwt")
    f = tmp_path / "best.pt"; f.write_bytes(b"PT")
    rec = _Recorder()
    lakefs_objects.upload(f, "poulaki", "main/checkpoints/exp/best.pt", opener=rec)
    method, url, auth, data = rec.calls[0]
    assert method == "POST" and auth == "Bearer jwt"
    assert url == "https://lakefs.example/api/v1/repositories/poulaki/branches/main/objects?path=checkpoints%2Fexp%2Fbest.pt"
    assert b"PT" in data


def test_download_streams_ref_object(tmp_path, monkeypatch):
    monkeypatch.setenv("LAKEFS_ENDPOINT", "https://lakefs.example")
    monkeypatch.setenv("LAKEFS_BEARER_TOKEN", "jwt")
    rec = _Recorder(b"weights")
    out = tmp_path / "w.pt"
    lakefs_objects.download("poulaki", "main/models/yolo.pt", out, opener=rec)
    assert out.read_bytes() == b"weights"
    assert rec.calls[0][1].endswith("/repositories/poulaki/refs/main/objects?path=models%2Fyolo.pt")


def test_staged_dataset_dir_resolution(tmp_path, monkeypatch):
    monkeypatch.delenv("KUBECORE_DATASET_DIR", raising=False)
    assert entry._staged_dataset_dir("c0ffee/") is None
    root = tmp_path / "commit"; (root / "dataset").mkdir(parents=True)
    monkeypatch.setenv("KUBECORE_DATASET_DIR", str(root))
    with pytest.raises(SystemExit):  # mounted but no dataset at the root -> loud, never S3
        entry._staged_dataset_dir("c0ffee/")
    # no legacy fallback: a dataset/ folder is not the dataset any more
    (root / "dataset" / "data.yaml").write_text("path: .\n")
    with pytest.raises(SystemExit):
        entry._staged_dataset_dir("c0ffee/")
    # the dataset repository root IS the dataset (root layout)
    (root / "data.yaml").write_text("path: .\n")
    assert entry._staged_dataset_dir("c0ffee/") == str(root)


def test_dataset_location_is_the_pinned_manifest_summary():
    summary = {"bucket": "dataset-cats", "prefix": "c0ffee/", "commit": "c0ffee",
               "ref": "dataset-cats@v2"}
    assert entry._dataset_location(summary) == ("dataset-cats", "c0ffee/", "c0ffee")
    # no dataset-loading output: no pinned commit to read -> stop, never guess
    with pytest.raises(SystemExit, match="no pinned dataset commit"):
        entry._dataset_location({})


def test_platform_token_enables_mlflow_without_machine_key(monkeypatch):
    monkeypatch.delenv("ZITADEL_MACHINE_KEY_FILE", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "jwt")
    monkeypatch.setenv("MLFLOW_TRACKING_AUTH", "zitadel")
    assert entry._setup_mlflow_auth() is True
    assert "MLFLOW_TRACKING_AUTH" not in entry.__dict__.get("os").environ
