"""Datasets as lakeFS repositories (novelcore/kaos#1283 F-07).

A dataset is its own repository, ``dataset-<name>``, with the YOLO layout at the
repository ROOT. A run picks ``dataset = <dataset-repo>@<version>`` (``vN`` tag
or ``latest`` = head of ``main``); dataset-loading pins the commit that resolves
to (CON-04) and every later step — in-cluster and on MeluXina — reads that
commit. Pipeline outputs go to the app's own ``outputs-<app>`` repository on
``main``, one commit per run.
"""
import contextlib
import importlib
import importlib.util
import io
import json
import sys
import urllib.request
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kubecore.enhance import enhance  # noqa: E402
from kubecore.meluxina import MELUXINA_SUBMIT_CODE  # noqa: E402

CTX_PATH = ROOT / "kubecore/local-dev/pipeline-context.yaml"


@contextlib.contextmanager
def _step_app(step):
    """Import a step's ``app`` package in isolation: every step names its
    package ``app``, and other test modules leave theirs in sys.modules."""
    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(ROOT / "steps" / step))
    try:
        yield
    finally:
        sys.path.remove(str(ROOT / "steps" / step))
        for k in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def _load_file(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _S3:
    """head_object / list_objects_v2 / list_buckets over a fixed key set (the
    lakeFS S3 gateway: bucket = repository)."""

    def __init__(self, keys):
        self.keys = set(keys)

    def head_object(self, Bucket, Key):
        if f"{Bucket}/{Key}" not in self.keys:
            raise KeyError(Key)
        return {}

    def list_objects_v2(self, Bucket, Prefix="", MaxKeys=1000, Delimiter=None):
        keys = [k[len(Bucket) + 1:] for k in self.keys if k.startswith(Bucket + "/")]
        hits = [k for k in keys if k.startswith(Prefix)]
        return {"KeyCount": len(hits), "Contents": [{"Key": k} for k in hits]}

    def list_buckets(self):
        return {"Buckets": [{"Name": n} for n in sorted({k.split("/", 1)[0] for k in self.keys})]}


# ---------------------------------------------------------------- parsing `<repo>@<version>`

@pytest.mark.parametrize("value,repo,version,ref", [
    ("dataset-default@latest", "dataset-default", "latest", "main"),   # latest = head of main
    ("dataset-cats@v1", "dataset-cats", "v1", "v1"),                   # a version is a tag
    ("dataset-night-sky@v12", "dataset-night-sky", "v12", "v12"),
    ("  dataset-cats@v2  ", "dataset-cats", "v2", "v2"),               # form whitespace
])
def test_parse_accepts_repo_at_version(value, repo, version, ref):
    with _step_app("dataset_loading"):
        parsed = importlib.import_module("app.models.dataset").parse_dataset_ref(value)
    assert (parsed.repo, parsed.version, parsed.ref) == (repo, version, ref)


MALFORMED = ["", "dataset-default", "dataset-default@", "@latest", "cats@v1",
             "dataset-cats@main", "dataset-cats@1", "dataset-cats@v1.2", "dataset-@v1",
             "dataset-Cats@v1", "dataset-cats@latest@v1", "dataset-cats @v1"]


@pytest.mark.parametrize("value", MALFORMED)
def test_parse_refuses_malformed_values_with_a_clear_message(value):
    with _step_app("dataset_loading"):
        with pytest.raises(ValueError, match=r"dataset must be '<dataset-repo>@<version>'.*dataset-default@latest"):
            importlib.import_module("app.models.dataset").parse_dataset_ref(value)


@pytest.mark.parametrize("value", MALFORMED)
def test_validation_refuses_malformed_values_too(value):
    mod, _ = _cv()
    with pytest.raises(mod.ConfigValidationError, match="dataset must be '<dataset-repo>@<version>'"):
        mod.parse_dataset_ref(value)


def test_validation_and_loading_parse_alike():
    mod, _ = _cv()
    with _step_app("dataset_loading"):
        loading = importlib.import_module("app.models.dataset").parse_dataset_ref
        for value in ["dataset-default@latest", "dataset-cats@v3"] + MALFORMED:
            try:
                want = tuple(loading(value))
            except ValueError:
                with pytest.raises(mod.ConfigValidationError):
                    mod.parse_dataset_ref(value)
            else:
                assert tuple(mod.parse_dataset_ref(value)) == want


# ---------------------------------------------------------------- dataset-loading

def _service():
    svc_mod = importlib.import_module("app.services.dataset_loading")
    svc = svc_mod.DatasetLoadingService.__new__(svc_mod.DatasetLoadingService)
    svc._logger = svc_mod.logging.getLogger("test")
    svc._lakefs_endpoint = "https://lakefs.example"
    svc._lakefs_access_key, svc._lakefs_secret_key = "a", "s"
    return svc_mod, svc


def _params(models, dataset="dataset-cats@latest", **kw):
    repo, _, version = dataset.partition("@")
    return models.YoloDatasetParams(version=dataset, source="lakefs", output_dir="/tmp/x",
                                    lakefs_repo=repo, lakefs_ref="main" if version == "latest" else version, **kw)


def _resolve(keys, dataset="dataset-cats@latest", commit="c0ffee"):
    with _step_app("dataset_loading"):
        models = importlib.import_module("app.models.dataset")
        _, svc = _service()
        svc._s3 = _S3(keys)
        return svc._resolve_source(_params(models, dataset), commit)


def test_loading_reads_the_pinned_commit_at_the_repository_root():
    assert _resolve({"dataset-cats/c0ffee/data.yaml"}) == ("dataset-cats", "c0ffee/")


def test_loading_reads_the_ref_when_no_commit_is_given():
    # (only reachable for a direct service call; the run always pins a commit)
    assert _resolve({"dataset-cats/v2/data.yaml"}, "dataset-cats@v2", commit=None) == ("dataset-cats", "v2/")


@pytest.mark.parametrize("keys", [
    set(),
    {"dataset-cats/c0ffee/dataset/data.yaml"},          # old {ref}/dataset/ layout: not read any more
    {"dataset-cats/c0ffee/dataset/cats/data.yaml"},     # old {ref}/dataset/<version>/ layout
    {"dataset-cats/c0ffee/images/train/a.jpg"},         # no data.yaml
])
def test_loading_has_no_legacy_layout_fallback(keys):
    with pytest.raises(Exception, match="no data.yaml at s3://dataset-cats/c0ffee/") as exc:
        _resolve(keys)
    assert type(exc.value).__name__ == "DatasetLoadingError"


class _Resp:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code} error")


def _pin(monkeypatch, dataset, resp):
    """Resolve ``dataset`` through the service's lakeFS call; returns (commit, urls)."""
    urls = []
    with _step_app("dataset_loading"):
        svc_mod, svc = _service()
        models = importlib.import_module("app.models.dataset")
        p = _params(models, dataset)

        def fake_get(url, params=None, auth=None, timeout=0):
            urls.append((url, params, auth))
            return resp

        monkeypatch.setattr(svc_mod.requests, "get", fake_get)
        try:
            return svc._fetch_lakefs_commit(p.lakefs_repo, p.lakefs_ref), urls
        except svc_mod.DatasetLoadingError as exc:
            return exc, urls


def test_latest_is_pinned_from_the_head_of_main(monkeypatch):
    commit, urls = _pin(monkeypatch, "dataset-cats@latest", _Resp(body={"results": [{"id": "c0ffee"}]}))
    assert commit == "c0ffee"
    url, params, auth = urls[0]
    assert url == "https://lakefs.example/api/v1/repositories/dataset-cats/refs/main/commits"
    assert params == {"amount": 1} and auth == ("a", "s")


def test_a_version_is_pinned_from_its_tag(monkeypatch):
    commit, urls = _pin(monkeypatch, "dataset-cats@v2", _Resp(body={"results": [{"id": "beef"}]}))
    assert commit == "beef"
    assert urls[0][0].endswith("/repositories/dataset-cats/refs/v2/commits")


@pytest.mark.parametrize("resp,message", [
    (_Resp(404), "dataset dataset-cats@v2 not found"),
    (_Resp(500), "could not resolve dataset-cats@v2"),
    (_Resp(body={"results": []}), "has no commits"),
])
def test_an_unresolvable_version_stops_the_run(monkeypatch, resp, message):
    """CON-04: never fall back to reading an unpinned ref."""
    err, _ = _pin(monkeypatch, "dataset-cats@v2", resp)
    assert type(err).__name__ == "DatasetLoadingError" and message in str(err)


def test_commit_lookup_uses_the_lakefs_page_size_param():
    src = (ROOT / "steps/dataset_loading/app/services/dataset_loading.py").read_text()
    assert 'params={"amount": 1}' in src and 'params={"limit": 1}' not in src


def test_loading_maps_the_dataset_param_onto_the_service_params():
    with _step_app("dataset_loading"):
        entry = importlib.import_module("app.entry")
        params = entry._resolve_dataset_params({"dataset": "dataset-cats@latest", "data": {"source": "lakefs"}})
        assert (params["lakefs_repo"], params["lakefs_ref"]) == ("dataset-cats", "main")
        assert params["version"] == "dataset-cats@latest"
        params = entry._resolve_dataset_params({"dataset": "dataset-cats@v7", "data": {"source": "lakefs"}})
        assert params["lakefs_ref"] == "v7"
        with pytest.raises(ValueError, match="dataset must be"):
            entry._resolve_dataset_params({"dataset": "main", "data": {}})


def test_loading_emits_dataset_and_pinned_commit(tmp_path, monkeypatch):
    with _step_app("dataset_loading"):
        entry = importlib.import_module("app.entry")
        monkeypatch.setattr(entry, "DATASET_DIR", str(tmp_path))
        monkeypatch.setattr(entry, "OUTPUT_DIR", tmp_path / "out")
        (tmp_path / "data.yaml").write_text("kpt_shape: [4, 3]\n")
        (tmp_path / "dataset_manifest.json").write_text(json.dumps(
            {"bucket": "dataset-cats", "prefix": "c0ffee/", "label_keys": {"train": ["x"]},
             "lakefs_commit": "c0ffee"}))
        entry._write_outputs({"dataset": "dataset-cats@v2"})
    summary = json.loads((tmp_path / "out" / "manifest-summary.json").read_text())
    assert summary == {"bucket": "dataset-cats", "prefix": "c0ffee/",
                       "label_keys": {"_present": True}, "ref": "dataset-cats@v2",
                       "repo": "dataset-cats", "version": "v2", "commit": "c0ffee"}


# ---------------------------------------------------------------- config-validation

def _cv():
    mod = _load_file("steps/config_validation/app/services/config_validation.py", "cv_service")
    return mod, mod.ConfigValidationService()


DATA = {"source": "lakefs"}


def test_validation_accepts_the_root_layout_at_the_picked_version():
    mod, svc = _cv()
    svc._check_dataset_path(_S3({"dataset-cats/main/data.yaml"}), DATA, mod.parse_dataset_ref("dataset-cats@latest"))
    svc._check_dataset_path(_S3({"dataset-cats/v2/data.yaml"}), DATA, mod.parse_dataset_ref("dataset-cats@v2"))


@pytest.mark.parametrize("key", ["dataset-cats/main/dataset/data.yaml", "dataset-cats/main/dataset/cats/data.yaml"])
def test_validation_does_not_accept_the_legacy_layouts(key):
    mod, svc = _cv()
    with pytest.raises(mod.ConfigValidationError, match="no data.yaml at its root"):
        svc._check_dataset_path(_S3({key}), DATA, mod.parse_dataset_ref("dataset-cats@latest"))


def test_validation_miss_names_the_dataset_repositories_that_exist():
    mod, svc = _cv()
    s3 = _S3({"dataset-default/main/data.yaml", "dataset-dogs/main/data.yaml", "outputs-yolo/main/x.pt"})
    with pytest.raises(mod.ConfigValidationError,
                       match="dataset not found or empty.*Dataset repositories: dataset-default, dataset-dogs"):
        svc._check_dataset_path(s3, DATA, mod.parse_dataset_ref("dataset-cats@latest"))


def test_validation_rejects_a_malformed_dataset_even_without_credentials(monkeypatch):
    mod, svc = _cv()
    for var in ("LAKEFS_ENDPOINT", "MLFLOW_TRACKING_URI"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(mod.ConfigValidationError, match="dataset must be '<dataset-repo>@<version>'"):
        svc.run({"dataset": "cats", "data": {}, "model": {}, "platform": {}})


# ---------------------------------------------------------------- outputs: outputs-<app>, main, one commit per run

def test_checkpoints_are_scoped_to_the_run(monkeypatch):
    with _step_app("model_training"):
        mt = importlib.import_module("app.services.model_training")
        params = type("P", (), {"checkpoint_prefix": "main/checkpoints/", "experiment_name": "exp"})()
        monkeypatch.setenv("KUBECORE_RUN_ID", "yolo-pipeline-x7k2p")
        assert mt.checkpoint_base(params) == "main/checkpoints/exp/yolo-pipeline-x7k2p"
        monkeypatch.delenv("KUBECORE_RUN_ID")
        assert mt.checkpoint_base(params) == "main/checkpoints/exp"


def test_training_commits_once_on_main_of_the_outputs_repo_naming_the_run(monkeypatch):
    with _step_app("model_training"):
        mt = importlib.import_module("app.services.model_training")
        svc = mt.TrainingService.__new__(mt.TrainingService)
        commits = []
        svc._lakefs_commit = lambda repo, branch, message, metadata=None: commits.append(
            (repo, branch, message, metadata))
        svc._get_mlflow_run_id = lambda: "mlflow-1"
        params = type("P", (), {
            "checkpoint_bucket": "outputs-yolo", "checkpoint_prefix": "main/checkpoints",
            "experiment_name": "exp", "dataset_repo": "dataset-cats", "dataset_version": "v2",
            "lakefs_commit": "c0ffee"})()
        monkeypatch.setenv("KUBECORE_RUN_ID", "yolo-pipeline-x7k2p")
        svc._commit_outputs(params)
    [(repo, branch, message, meta)] = commits
    assert (repo, branch) == ("outputs-yolo", "main")
    assert "yolo-pipeline-x7k2p" in message
    assert meta["dataset_ref"] == "dataset-cats@v2" and meta["dataset_commit"] == "c0ffee"


@pytest.mark.parametrize("rel", [
    "steps/model_quantization/app/services/quantization_service.py",
    "steps/qat_finetune/app/services/qat_service.py",
])
def test_quantization_outputs_commit_names_the_run(rel):
    src = (ROOT / rel).read_text()
    assert 'run = os.environ.get("KUBECORE_RUN_ID", "").strip()' in src
    assert '{run}".rstrip()' in src


def test_training_provenance_logs_dataset_repo_version_and_commit():
    src = (ROOT / "steps/model_training/app/services/model_training.py").read_text()
    assert '"dataset.repo": params.dataset_repo' in src
    assert '"dataset.version": params.dataset_version' in src
    assert '"dataset.lakefs_commit": params.lakefs_commit' in src


# ---------------------------------------------------------------- MeluXina

def _submit_globals(monkeypatch, step_command, **env):
    base = {"SLURM_TOKEN": "t", "WF_UID": "u", "STEP_NAME": "s", "IMAGE_REF": "r/x:t",
            "STEP_COMMAND": json.dumps(step_command)}
    for k, v in {**base, **env}.items():
        monkeypatch.setenv(k, v)
    g = {}
    exec(MELUXINA_SUBMIT_CODE.split("jid = None")[0], g)
    return g


def test_hpc_pins_the_commit_dataset_loading_resolved(monkeypatch):
    summary = json.dumps({"bucket": "dataset-cats", "prefix": "c0ffee/", "commit": "c0ffee"})
    g = _submit_globals(monkeypatch, ["python", "-m", "train", "--manifest-summary", summary])
    assert g["pinned_commit"]() == "c0ffee"


def test_hpc_pins_the_commit_training_used(monkeypatch):
    tr = json.dumps({"best_checkpoint_s3": "s3://x", "dataset_commit": "c0ffee"})
    g = _submit_globals(monkeypatch, ["python", "-m", "qat", "--training-result", tr])
    assert g["pinned_commit"]() == "c0ffee"


def test_hpc_without_a_pinned_commit_resolves_the_ref(monkeypatch):
    g = _submit_globals(monkeypatch, ["python", "-m", "train", "--manifest-summary", "{}"])
    assert g["pinned_commit"]() == ""


@pytest.mark.parametrize("dataset,coords", [
    ("dataset-cats@latest", ("dataset-cats", "main")),
    ("dataset-cats@v3", ("dataset-cats", "v3")),
    ("", ("", "")),
])
def test_hpc_reads_repo_and_ref_from_the_dataset_param(monkeypatch, dataset, coords):
    g = _submit_globals(monkeypatch, ["python"], DATASET=dataset)
    assert g["dataset_coords"]() == coords


def test_hpc_refuses_a_malformed_dataset_param(monkeypatch, capsys):
    g = _submit_globals(monkeypatch, ["python"], DATASET="main")
    with pytest.raises(SystemExit):
        g["dataset_coords"]()
    assert "dataset must be <dataset-repo>@<version>" in capsys.readouterr().out


class _LakeFS:
    """urlopen stand-in for the lakeFS objects API the stage-in/out scripts use."""

    def __init__(self, objects):
        self.objects, self.urls = objects, []

    def __call__(self, req, timeout=0):
        url = req.full_url if hasattr(req, "full_url") else req
        self.urls.append(url)
        if "/commits" in url:
            return io.BytesIO(json.dumps({"results": [{"id": "tipcommit"}]}).encode())
        if "/objects/ls?" in url:
            q = dict(p.split("=", 1) for p in url.split("?", 1)[1].split("&"))
            prefix = urllib.request.unquote(q.get("prefix", ""))
            res = [{"path": p, "path_type": "object", "size_bytes": len(b)}
                   for p, b in self.objects.items() if p.startswith(prefix)]
            return io.BytesIO(json.dumps({"results": res, "pagination": {"has_more": False}}).encode())
        if "/objects?" in url:
            path = urllib.request.unquote(url.split("path=", 1)[1])
            return io.BytesIO(self.objects.get(path, b""))
        return io.BytesIO(b"{}")


def _stage_globals():
    g = {}
    exec(MELUXINA_SUBMIT_CODE.split("API = ")[0], g)
    return g


def _stage_env(monkeypatch, tmp_path, **extra):
    for k, v in {"LAKEFS_ENDPOINT": "https://lakefs.example", "LAKEFS_BEARER_TOKEN": "jwt",
                 "DATASET_REPO": "dataset-cats", "DATASET_REF": "v2",
                 "DATASET_CACHE": str(tmp_path / "cache"),
                 "DATASET_DIRFILE": str(tmp_path / "dir"), **extra}.items():
        monkeypatch.setenv(k, v)


def test_stage_in_downloads_the_root_layout_of_the_pinned_commit(tmp_path, monkeypatch):
    lakefs = _LakeFS({"data.yaml": b"y", "images/train/a.jpg": b"img", "labels/train/a.txt": b"l",
                      "README.md": b"notes", "dataset/data.yaml": b"legacy"})
    monkeypatch.setattr(urllib.request, "urlopen", lakefs)
    _stage_env(monkeypatch, tmp_path, DATASET_COMMIT="c0ffee")
    exec(_stage_globals()["STAGEIN"], {})
    dest = tmp_path / "cache" / "dataset-cats" / "c0ffee"   # keyed by (repository, commit)
    assert (dest / "data.yaml").read_bytes() == b"y"
    assert (dest / "images/train/a.jpg").exists() and (dest / "labels/train/a.txt").exists()
    assert not (dest / "README.md").exists() and not (dest / "dataset").exists()
    assert not any("/commits" in u for u in lakefs.urls), "a pinned commit is never re-resolved"
    assert all("/repositories/dataset-cats/" in u for u in lakefs.urls)
    assert (tmp_path / "dir").read_text() == str(dest)


def test_stage_in_resolves_the_version_ref_when_nothing_was_pinned(tmp_path, monkeypatch):
    lakefs = _LakeFS({"data.yaml": b"y", "images/a.jpg": b"img"})
    monkeypatch.setattr(urllib.request, "urlopen", lakefs)
    _stage_env(monkeypatch, tmp_path)
    monkeypatch.delenv("DATASET_COMMIT", raising=False)
    exec(_stage_globals()["STAGEIN"], {})
    assert (tmp_path / "cache" / "dataset-cats" / "tipcommit" / "data.yaml").exists()
    assert any("/refs/v2/commits" in u for u in lakefs.urls)


def test_stage_out_goes_to_the_apps_outputs_repository(tmp_path, monkeypatch):
    lakefs = _LakeFS({})
    monkeypatch.setattr(urllib.request, "urlopen", lakefs)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "training-result.json").write_text("{}")
    for k, v in {"LAKEFS_ENDPOINT": "https://lakefs.example", "LAKEFS_BEARER_TOKEN": "jwt",
                 "OUTPUTS_REPO": "outputs-yolo", "OUTPUTS_REF": "main",
                 "WF_UID": "u", "STEP_NAME": "model-training", "STEP_OUTPUTS": "training-result",
                 "KAOS_WORK": str(tmp_path)}.items():
        monkeypatch.setenv(k, v)
    g = {}
    exec(MELUXINA_SUBMIT_CODE.split("import base64, json")[0], g)
    exec(g["STAGEOUT"], {})
    assert any("/repositories/outputs-yolo/branches/main/objects?" in u for u in lakefs.urls), lakefs.urls
    assert not any("dataset-" in u for u in lakefs.urls), "outputs never touch a dataset repository"


def _raw():
    return {
        "apiVersion": "argoproj.io/v1alpha1", "kind": "WorkflowTemplate", "metadata": {"name": "x"},
        "spec": {"entrypoint": "p",
                 "arguments": {"parameters": [{"name": "dataset", "value": "dataset-default@latest"}]},
                 "templates": [
            {"name": "p", "dag": {"tasks": [{"name": "model-training", "template": "model-training"}]}},
            {"name": "model-training", "container": {"image": "i", "command": ["python", "-m", "train"],
                                                     "resources": {"requests": {"nvidia.com/gpu": 1}}}},
        ]},
    }


def test_platform_wiring_points_outputs_at_the_apps_outputs_repository():
    ctx = yaml.safe_load(CTX_PATH.read_text())
    ctx["app"] = "yolo"
    ctx["lakefs"]["outputsBranch"] = "outputs"          # legacy context key: ignored now
    ctx["checkpoints"] = {"bucket": "poulaki", "prefix": "outputs/checkpoints"}
    out = enhance(_raw(), ctx)
    step = next(t for t in out["spec"]["templates"] if t["name"] == "model-training")
    env = {e["name"]: e.get("value") for e in step["container"]["env"]}
    assert env["KUBECORE_RUN_ID"] == "{{workflow.name}}"
    assert env["CHECKPOINT_BUCKET"] == "outputs-yolo"
    assert env["CHECKPOINT_PREFIX"] == "main/checkpoints"
    mr = next(t for t in out["spec"]["templates"] if t["name"] == "meluxina-run")
    menv = {e["name"]: e.get("value") for e in mr["container"]["env"]}
    assert menv["OUTPUTS_REPO"] == "outputs-yolo" and menv["OUTPUTS_REF"] == "main"
    assert menv["CHECKPOINT_BUCKET"] == "outputs-yolo" and menv["CHECKPOINT_PREFIX"] == "main/checkpoints"
    assert menv["KUBECORE_RUN_ID"] == "{{workflow.name}}"
    assert "OUTPUTS_BRANCH" not in menv
    assert "OUTPUTS_BRANCH" not in MELUXINA_SUBMIT_CODE


def test_platform_group_override_carries_the_outputs_repository():
    ctx = yaml.safe_load(CTX_PATH.read_text())
    ctx["app"] = "yolo"
    raw = _raw()
    raw["spec"]["templates"].append({"name": "compose-and-validate", "container": {"image": "i", "args": []}})
    out = enhance(raw, ctx)
    compose = next(t for t in out["spec"]["templates"] if t["name"] == "compose-and-validate")
    assert "platform.checkpoints.bucket=outputs-yolo" in compose["container"]["args"]
    assert "platform.checkpoints.prefix=main/checkpoints" in compose["container"]["args"]


# ---------------------------------------------------------------- review fixes

@pytest.mark.parametrize("step", ["qat_finetune", "model_quantization"])
def test_calibration_download_takes_only_the_dataset(step, tmp_path, monkeypatch):
    """Other files at the dataset repository root (README, notes) must not be
    downloaded on every run: only data.yaml, images/ and labels/."""
    entry = _load_file(f"steps/{step}/app/entry.py", f"{step}_entry_dl")
    keys = ["c0ffee/data.yaml", "c0ffee/images/train/a.jpg", "c0ffee/labels/train/a.txt",
            "c0ffee/README.md", "c0ffee/notes/x.txt"]
    got = []

    class _Boto:
        def get_paginator(self, _):
            return type("P", (), {"paginate": lambda s, **k: [{"Contents": [{"Key": x} for x in keys]}]})()

        def download_file(self, bucket, key, local):
            got.append(key)

    monkeypatch.setenv("LAKEFS_ACCESS_KEY", "a")
    monkeypatch.setenv("LAKEFS_SECRET_KEY", "s")
    monkeypatch.setitem(sys.modules, "boto3", type("B", (), {"client": staticmethod(lambda *a, **k: _Boto())}))
    entry._download_prefix("dataset-cats", "c0ffee/", str(tmp_path))
    assert sorted(got) == sorted(keys[:3])


@pytest.mark.parametrize("rel", ["steps/model_training/app/entry.py", "steps/qat_finetune/app/entry.py",
                                 "steps/model_quantization/app/entry.py"])
def test_an_override_outside_the_layout_is_not_silently_replaced_off_cluster(rel, tmp_path, monkeypatch):
    with _step_app(rel.split("/")[1]):
        entry = _load_file(rel, rel.replace("/", "_"))
        (tmp_path / "data.yaml").write_text("path: .\n")
        monkeypatch.setenv("KUBECORE_DATASET_DIR", str(tmp_path))
        with pytest.raises(SystemExit, match="not staged off-cluster"):
            entry._staged_dataset_dir("c0ffee/custom/ds/")


def test_slurm_env_omits_unset_checkpoint_settings():
    """Step code reads os.environ.get(KEY, default): an empty value would beat
    its default (old code then uploaded to bucket '')."""
    src = MELUXINA_SUBMIT_CODE
    assert "'CHECKPOINT_BUCKET=' +" not in src
    assert "if os.environ.get(k)]" in src
