"""PRD-1016 F-04/F-05 for qat-finetune and model-quantization off-cluster
(MeluXina): the staged dataset is used in place of the S3-gateway download,
lakeFS I/O goes through the objects API with the bearer token, and a missing
credential path fails loudly (live job 5154708: boto3 NoCredentialsError)."""
import ast
import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STEPS = ("qat_finetune", "model_quantization")


def _load(step, rel, name):
    spec = importlib.util.spec_from_file_location(f"{step}_{name}", ROOT / "steps" / step / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Recorder:
    def __init__(self, body=b"weights"):
        self.calls, self.body = [], body

    def __call__(self, req, timeout=0):
        self.calls.append((req.get_method(), req.full_url, req.headers.get("Authorization"), req.data))
        return io.BytesIO(self.body)


@pytest.mark.parametrize("step", STEPS)
def test_bearer_client_speaks_the_two_boto3_verbs(step, tmp_path, monkeypatch):
    lo = _load(step, "app/services/lakefs_objects.py", "lakefs_objects")
    monkeypatch.setenv("LAKEFS_ENDPOINT", "https://lakefs.example")
    monkeypatch.setenv("LAKEFS_BEARER_TOKEN", "jwt")
    rec = _Recorder(b"PT")
    client = lo.BearerClient(opener=rec)
    out = tmp_path / "best.pt"
    client.download_file("poulaki", "main/checkpoints/exp/best.pt", str(out))
    assert out.read_bytes() == b"PT"
    assert rec.calls[0][1].endswith("/repositories/poulaki/refs/main/objects?path=checkpoints%2Fexp%2Fbest.pt")
    client.upload_file(str(out), "poulaki", "main/quantization/model_int8.tflite")
    method, url, auth, data = rec.calls[1]
    assert method == "POST" and auth == "Bearer jwt" and b"PT" in data
    assert url.endswith("/repositories/poulaki/branches/main/objects?path=quantization%2Fmodel_int8.tflite")


@pytest.mark.parametrize("step", STEPS)
def test_manager_builds_bearer_client_without_gateway_keys(step):
    """Wiring pin (importing the manager pulls torch): the bearer branch sits
    in _build_s3_client between the lakeFS-keys branch and the cloud chain."""
    tree = ast.parse((ROOT / "steps" / step / "app/manager.py").read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_build_s3_client")
    src = ast.unparse(fn)
    keys = src.index("cfg.lakefs_access_key")
    bearer = src.index("lakefs_objects.bearer_available()")
    cloud = src.index("cfg.s3_endpoint_url")
    assert keys < bearer < cloud
    assert "lakefs_objects.BearerClient()" in src


@pytest.mark.parametrize("step", STEPS)
def test_staged_dataset_is_linked_not_downloaded(step, tmp_path, monkeypatch):
    entry = _load(step, "app/entry.py", "entry")
    monkeypatch.delenv("KUBECORE_DATASET_DIR", raising=False)
    assert entry._staged_dataset_dir() is None
    root = tmp_path / "ref"; ds = root / "dataset" / "main"; (ds / "images" / "train").mkdir(parents=True)
    monkeypatch.setenv("KUBECORE_DATASET_DIR", str(root))
    monkeypatch.setenv("KUBECORE_DATASET_VERSION", "main")
    with pytest.raises(SystemExit):  # mounted but no dataset -> loud, never S3
        entry._staged_dataset_dir()
    (ds / "data.yaml").write_text("path: /uploader/machine\ntrain: images/train\n")
    assert entry._staged_dataset_dir() == str(ds)
    dest = tmp_path / "work"; dest.mkdir()
    entry._link_staged(str(ds), str(dest))
    assert (dest / "images").is_symlink() and (dest / "images/train").is_dir()
    assert not (dest / "data.yaml").is_symlink()  # writable copy for _fix_data_yaml
    entry._fix_data_yaml(str(dest))
    assert f"path: {dest}" in (dest / "data.yaml").read_text()
    assert (ds / "data.yaml").read_text().startswith("path: /uploader")  # staged copy untouched


@pytest.mark.parametrize("step", STEPS)
def test_gateway_download_without_keys_fails_loudly(step, tmp_path, monkeypatch):
    entry = _load(step, "app/entry.py", "entry")
    for var in ("LAKEFS_ACCESS_KEY", "LAKEFS_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(SystemExit, match="no lakeFS S3 keys"):
        entry._download_prefix("poulaki", "main/dataset/main/", str(tmp_path))
