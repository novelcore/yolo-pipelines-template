"""Datasets as lakeFS branches (yolo-pipelines-template#33).

A dataset lives at ``s3://{repo}/{ref}/dataset/`` — the same path on every
branch — and a run reads ONE commit: dataset-loading resolves the ref, every
later step (in-cluster and on MeluXina) reads that commit. Datasets uploaded in
the old ``{ref}/dataset/{ref}/`` layout keep working. Pipeline outputs go to the
project's ``outputs`` branch, under a per-run path.
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
    """head_object / list_objects_v2 over a fixed key set (the lakeFS S3 gateway)."""

    def __init__(self, keys):
        self.keys = set(keys)

    def head_object(self, Bucket, Key):
        if f"{Bucket}/{Key}" not in self.keys:
            raise KeyError(Key)
        return {}

    def list_objects_v2(self, Bucket, Prefix="", MaxKeys=1000, Delimiter=None):
        keys = [k[len(Bucket) + 1:] for k in self.keys if k.startswith(Bucket + "/")]
        if Delimiter:
            refs = sorted({Prefix + k[len(Prefix):].split("/", 1)[0] + "/" for k in keys
                           if k.startswith(Prefix) and "/" in k[len(Prefix):]})
            return {"CommonPrefixes": [{"Prefix": r} for r in refs]}
        hits = [k for k in keys if k.startswith(Prefix)]
        return {"KeyCount": len(hits), "Contents": [{"Key": k} for k in hits]}


# ---------------------------------------------------------------- dataset-loading

def _resolve(keys, branch="cats", commit="c0ffee"):
    with _step_app("dataset_loading"):
        svc_mod = importlib.import_module("app.services.dataset_loading")
        models = importlib.import_module("app.models.dataset")
        svc = svc_mod.DatasetLoadingService.__new__(svc_mod.DatasetLoadingService)
        svc._s3 = _S3(keys)
        svc._logger = svc_mod.logging.getLogger("test")
        params = models.YoloDatasetParams(version=branch, source="lakefs", output_dir="/tmp/x",
                                          lakefs_repo="poulaki", lakefs_branch=branch)
        return svc._resolve_source(params, commit)


def test_loading_reads_the_commit_at_the_standard_path():
    assert _resolve({"poulaki/c0ffee/dataset/data.yaml"}) == ("poulaki", "c0ffee/dataset/")


def test_loading_falls_back_to_the_old_layout():
    assert _resolve({"poulaki/c0ffee/dataset/cats/data.yaml"}) == ("poulaki", "c0ffee/dataset/cats/")


def test_loading_prefers_the_standard_path_over_the_old_one():
    keys = {"poulaki/c0ffee/dataset/data.yaml", "poulaki/c0ffee/dataset/cats/data.yaml"}
    assert _resolve(keys) == ("poulaki", "c0ffee/dataset/")


def test_loading_falls_back_to_the_one_explicit_version_folder():
    # Uploaded with --data-version v2 before the version folder was dropped.
    assert _resolve({"poulaki/c0ffee/dataset/v2/data.yaml",
                     "poulaki/c0ffee/dataset/v2/images/a.jpg"}) == ("poulaki", "c0ffee/dataset/v2/")


def test_loading_refuses_to_guess_between_version_folders():
    with pytest.raises(Exception, match="2 old-layout versions.*dataset/v1/.*dataset/v2/") as exc:
        _resolve({"poulaki/c0ffee/dataset/v1/data.yaml", "poulaki/c0ffee/dataset/v2/data.yaml"})
    assert type(exc.value).__name__ == "DatasetLoadingError"


def test_loading_reads_the_ref_when_no_commit_resolved():
    assert _resolve({"poulaki/cats/dataset/data.yaml"}, commit=None) == ("poulaki", "cats/dataset/")


def test_commit_lookup_uses_the_lakefs_page_size_param():
    src = (ROOT / "steps/dataset_loading/app/services/dataset_loading.py").read_text()
    assert 'params={"amount": 1}' in src and 'params={"limit": 1}' not in src


def test_loading_emits_ref_and_commit(tmp_path, monkeypatch):
    with _step_app("dataset_loading"):
        entry = importlib.import_module("app.entry")
        monkeypatch.setattr(entry, "DATASET_DIR", str(tmp_path))
        monkeypatch.setattr(entry, "OUTPUT_DIR", tmp_path / "out")
        (tmp_path / "data.yaml").write_text("kpt_shape: [4, 3]\n")
        (tmp_path / "dataset_manifest.json").write_text(json.dumps(
            {"bucket": "poulaki", "prefix": "c0ffee/dataset/", "label_keys": {"train": ["x"]},
             "lakefs_commit": "c0ffee"}))
        entry._write_outputs({"data": {"ref": "cats"}})
    summary = json.loads((tmp_path / "out" / "manifest-summary.json").read_text())
    assert summary == {"bucket": "poulaki", "prefix": "c0ffee/dataset/",
                       "label_keys": {"_present": True}, "ref": "cats", "commit": "c0ffee"}


# ---------------------------------------------------------------- config-validation

def _cv():
    mod = _load_file("steps/config_validation/app/services/config_validation.py", "cv_service")
    return mod, mod.ConfigValidationService()


DATA = {"ref": "cats", "source": "lakefs"}
PLATFORM = {"lakefs": {"repository": "poulaki"}}


def test_validation_accepts_both_layouts():
    _, svc = _cv()
    svc._check_dataset_path(_S3({"poulaki/cats/dataset/data.yaml"}), DATA, PLATFORM)
    svc._check_dataset_path(_S3({"poulaki/cats/dataset/cats/data.yaml"}), DATA, PLATFORM)


def test_validation_accepts_one_explicit_version_folder():
    _, svc = _cv()
    svc._check_dataset_path(_S3({"poulaki/cats/dataset/v2/data.yaml"}), DATA, PLATFORM)


def test_validation_refuses_to_guess_between_version_folders():
    mod, svc = _cv()
    s3 = _S3({"poulaki/cats/dataset/v1/data.yaml", "poulaki/cats/dataset/v2/data.yaml"})
    with pytest.raises(mod.ConfigValidationError, match="2 old-layout versions.*path_override"):
        svc._check_dataset_path(s3, DATA, PLATFORM)


def test_validation_miss_names_the_datasets_that_exist():
    mod, svc = _cv()
    s3 = _S3({"poulaki/main/dataset/main/data.yaml", "poulaki/dogs/dataset/data.yaml",
              "poulaki/outputs/checkpoints/x.pt"})
    with pytest.raises(mod.ConfigValidationError, match="Datasets in this repository: dogs, main"):
        svc._check_dataset_path(s3, DATA, PLATFORM)


# ---------------------------------------------------------------- model-training

def test_checkpoints_are_scoped_to_the_run(monkeypatch):
    with _step_app("model_training"):
        mt = importlib.import_module("app.services.model_training")
        params = type("P", (), {"checkpoint_prefix": "outputs/checkpoints/", "experiment_name": "exp"})()
        monkeypatch.setenv("KUBECORE_RUN_ID", "yolo-pipeline-x7k2p")
        assert mt.checkpoint_base(params) == "outputs/checkpoints/exp/yolo-pipeline-x7k2p"
        monkeypatch.delenv("KUBECORE_RUN_ID")
        assert mt.checkpoint_base(params) == "outputs/checkpoints/exp"


# ---------------------------------------------------------------- MeluXina

def _submit_globals(monkeypatch, step_command):
    for k, v in {"SLURM_TOKEN": "t", "WF_UID": "u", "STEP_NAME": "s", "IMAGE_REF": "r/x:t",
                 "STEP_COMMAND": json.dumps(step_command)}.items():
        monkeypatch.setenv(k, v)
    g = {}
    exec(MELUXINA_SUBMIT_CODE.split("jid = None")[0], g)
    return g


def test_hpc_pins_the_commit_dataset_loading_resolved(monkeypatch):
    summary = json.dumps({"bucket": "poulaki", "prefix": "c0ffee/dataset/", "commit": "c0ffee"})
    g = _submit_globals(monkeypatch, ["python", "-m", "train", "--manifest-summary", summary])
    assert g["pinned_commit"]() == "c0ffee"


def test_hpc_pins_the_commit_training_used(monkeypatch):
    tr = json.dumps({"best_checkpoint_s3": "s3://x", "dataset_commit": "c0ffee"})
    g = _submit_globals(monkeypatch, ["python", "-m", "qat", "--training-result", tr])
    assert g["pinned_commit"]() == "c0ffee"


def test_hpc_without_a_pinned_commit_resolves_the_ref(monkeypatch):
    g = _submit_globals(monkeypatch, ["python", "-m", "train", "--manifest-summary", "{}"])
    assert g["pinned_commit"]() == ""


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


def test_stage_in_downloads_only_the_pinned_dataset(tmp_path, monkeypatch):
    lakefs = _LakeFS({"dataset/data.yaml": b"y", "dataset/images/a.jpg": b"img",
                      "checkpoints/old.pt": b"weights"})
    monkeypatch.setattr(urllib.request, "urlopen", lakefs)
    for k, v in {"LAKEFS_ENDPOINT": "https://lakefs.example", "LAKEFS_BEARER_TOKEN": "jwt",
                 "DATASET_REPO": "poulaki", "DATASET_REF": "cats", "DATASET_COMMIT": "c0ffee",
                 "DATASET_CACHE": str(tmp_path / "cache"),
                 "DATASET_DIRFILE": str(tmp_path / "dir")}.items():
        monkeypatch.setenv(k, v)
    exec(_stage_globals()["STAGEIN"], {})
    dest = tmp_path / "cache" / "poulaki" / "c0ffee"
    assert (dest / "dataset/data.yaml").read_bytes() == b"y"
    assert not (dest / "checkpoints").exists()
    assert not any("/commits" in u for u in lakefs.urls), "a pinned commit is never re-resolved"


def test_stage_in_takes_the_whole_ref_for_pre_layout_data(tmp_path, monkeypatch):
    lakefs = _LakeFS({"data.yaml": b"y", "images/a.jpg": b"img"})
    monkeypatch.setattr(urllib.request, "urlopen", lakefs)
    for k, v in {"LAKEFS_ENDPOINT": "https://lakefs.example", "LAKEFS_BEARER_TOKEN": "jwt",
                 "DATASET_REPO": "poulaki", "DATASET_REF": "cats",
                 "DATASET_CACHE": str(tmp_path / "cache"),
                 "DATASET_DIRFILE": str(tmp_path / "dir")}.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("DATASET_COMMIT", raising=False)
    exec(_stage_globals()["STAGEIN"], {})
    assert (tmp_path / "cache" / "poulaki" / "tipcommit" / "data.yaml").exists()


def test_stage_out_goes_to_the_outputs_branch(tmp_path, monkeypatch):
    lakefs = _LakeFS({})
    monkeypatch.setattr(urllib.request, "urlopen", lakefs)
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "training-result.json").write_text("{}")
    for k, v in {"LAKEFS_ENDPOINT": "https://lakefs.example", "LAKEFS_BEARER_TOKEN": "jwt",
                 "DATASET_REPO": "poulaki", "DATASET_REF": "cats", "OUTPUTS_BRANCH": "outputs",
                 "WF_UID": "u", "STEP_NAME": "model-training", "STEP_OUTPUTS": "training-result",
                 "KAOS_WORK": str(tmp_path)}.items():
        monkeypatch.setenv(k, v)
    g = {}
    exec(MELUXINA_SUBMIT_CODE.split("import base64, json")[0], g)
    exec(g["STAGEOUT"], {})
    assert any("/branches/outputs/objects?" in u for u in lakefs.urls), lakefs.urls


def _raw():
    return {
        "apiVersion": "argoproj.io/v1alpha1", "kind": "WorkflowTemplate", "metadata": {"name": "x"},
        "spec": {"entrypoint": "p", "arguments": {"parameters": [{"name": "data-ref", "value": "main"}]},
                 "templates": [
            {"name": "p", "dag": {"tasks": [{"name": "model-training", "template": "model-training"}]}},
            {"name": "model-training", "container": {"image": "i", "command": ["python", "-m", "train"],
                                                     "resources": {"requests": {"nvidia.com/gpu": 1}}}},
        ]},
    }


def test_platform_wiring_follows_the_outputs_branch():
    ctx = yaml.safe_load(CTX_PATH.read_text())
    ctx["lakefs"]["outputsBranch"] = "outputs"
    ctx["checkpoints"] = {"bucket": "poulaki", "prefix": "outputs/checkpoints"}
    out = enhance(_raw(), ctx)
    step = next(t for t in out["spec"]["templates"] if t["name"] == "model-training")
    env = {e["name"]: e.get("value") for e in step["container"]["env"]}
    assert env["KUBECORE_RUN_ID"] == "{{workflow.name}}"
    assert env["CHECKPOINT_PREFIX"] == "outputs/checkpoints"
    mr = next(t for t in out["spec"]["templates"] if t["name"] == "meluxina-run")
    menv = {e["name"]: e.get("value") for e in mr["container"]["env"]}
    assert menv["OUTPUTS_BRANCH"] == "outputs"
    assert menv["CHECKPOINT_PREFIX"] == "outputs/checkpoints"
    assert menv["KUBECORE_RUN_ID"] == "{{workflow.name}}"


def test_no_outputs_branch_keeps_todays_hand_off():
    ctx = yaml.safe_load(CTX_PATH.read_text())
    ctx["lakefs"].pop("outputsBranch", None)
    mr = next(t for t in enhance(_raw(), ctx)["spec"]["templates"] if t["name"] == "meluxina-run")
    assert {e["name"]: e.get("value") for e in mr["container"]["env"]}["OUTPUTS_BRANCH"] == ""


# ---------------------------------------------------------------- review fixes

@pytest.mark.parametrize("step", ["qat_finetune", "model_quantization"])
def test_calibration_download_takes_only_the_dataset(step, tmp_path, monkeypatch):
    """An old-layout copy kept under the same prefix (dataset/<ref>/) must not
    be downloaded on every run."""
    entry = _load_file(f"steps/{step}/app/entry.py", f"{step}_entry_dl")
    keys = ["c0ffee/dataset/data.yaml", "c0ffee/dataset/images/train/a.jpg",
            "c0ffee/dataset/labels/train/a.txt", "c0ffee/dataset/cats/images/train/a.jpg"]
    got = []

    class _Boto:
        def get_paginator(self, _):
            return type("P", (), {"paginate": lambda s, **k: [{"Contents": [{"Key": x} for x in keys]}]})()

        def download_file(self, bucket, key, local):
            got.append(key)

    monkeypatch.setenv("LAKEFS_ACCESS_KEY", "a")
    monkeypatch.setenv("LAKEFS_SECRET_KEY", "s")
    monkeypatch.setitem(sys.modules, "boto3", type("B", (), {"client": staticmethod(lambda *a, **k: _Boto())}))
    entry._download_prefix("poulaki", "c0ffee/dataset/", str(tmp_path))
    assert sorted(got) == sorted(keys[:3])


@pytest.mark.parametrize("rel", ["steps/model_training/app/entry.py", "steps/qat_finetune/app/entry.py",
                                 "steps/model_quantization/app/entry.py"])
def test_an_override_outside_the_layout_is_not_silently_replaced_off_cluster(rel, tmp_path, monkeypatch):
    with _step_app(rel.split("/")[1]):
        entry = _load_file(rel, rel.replace("/", "_"))
        (tmp_path / "dataset").mkdir()
        (tmp_path / "dataset" / "data.yaml").write_text("path: .\n")
        monkeypatch.setenv("KUBECORE_DATASET_DIR", str(tmp_path))
        with pytest.raises(SystemExit, match="not staged off-cluster"):
            entry._staged_dataset_dir("cats/custom/ds/", "cats")


def test_slurm_env_omits_unset_checkpoint_settings():
    """Step code reads os.environ.get(KEY, default): an empty value would beat
    its default (old code then uploaded to bucket '')."""
    src = MELUXINA_SUBMIT_CODE
    assert "'CHECKPOINT_BUCKET=' +" not in src
    assert "if os.environ.get(k)]" in src
