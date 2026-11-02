"""PRD novelcore/kaos#1283 F-07 / F-14: the steps read datasets and write results through kaos-lib (kaos_lib).

A run picks `<dataset-repo>@<version>`; dataset-loading resolves it to ONE commit; training, qat and
quantization read that commit (consumer) and write to `outputs-<app>` (write_run).
"""
import json
import re
from pathlib import Path

import pytest
import yaml
from conftest import DATA_YAML, ROOT, import_in, step_app
from kaos_lib.errors import DatasetError

# What the package refuses. Any repository name and any ref (vN, latest, a branch, a tag, a commit id)
# are valid picks since package round 3; the run still pins whatever it is to one commit.
MALFORMED = ["", "dataset-default", "dataset-default@", "@latest", "dataset-Cats@v1",
             "dataset-cats@latest@v1", "dataset-cats @v1"]


# ---------------------------------------------------------------- dataset-loading

def test_should_pin_latest_to_the_head_of_main_and_a_version_to_its_tag(client, fake):
    entry = import_in("dataset_loading", "app.entry")
    cfg = {"data": {"source": "lakefs"}}
    head = fake.head("dataset-cats")
    p = entry._resolve_dataset_params({**cfg, "dataset": "dataset-cats@latest"}, client)
    assert (p["lakefs_repo"], p["lakefs_ref"], p["version"]) == ("dataset-cats", head, "dataset-cats@latest")
    p = entry._resolve_dataset_params({**cfg, "dataset": "dataset-cats@v1"}, client)
    assert p["lakefs_ref"] == head, "v1 is a tag on that commit"


def test_should_stop_the_run_when_the_version_does_not_exist(client):
    entry = import_in("dataset_loading", "app.entry")
    with pytest.raises(DatasetError, match="no version v7.*Available: v1"):
        entry._resolve_dataset_params({"dataset": "dataset-cats@v7", "data": {"source": "lakefs"}}, client)
    with pytest.raises(DatasetError, match="does not exist"):
        entry._resolve_dataset_params({"dataset": "dataset-dogs@v1", "data": {"source": "lakefs"}}, client)


@pytest.mark.parametrize("value", MALFORMED)
def test_should_refuse_a_malformed_dataset_with_a_clear_message(client, value):
    entry = import_in("dataset_loading", "app.entry")
    with pytest.raises(DatasetError):
        entry._resolve_dataset_params({"dataset": value, "data": {"source": "lakefs"}}, client)


def test_should_not_resolve_a_commit_for_an_s3_override(client):
    entry = import_in("dataset_loading", "app.entry")
    p = entry._resolve_dataset_params(
        {"dataset": "dataset-cats@latest", "data": {"source": "lakefs", "path_override": "s3://b/p/"}}, client)
    assert "lakefs_ref" not in p and p["path_override"] == "s3://b/p/"


def test_should_emit_dataset_and_the_commit_every_later_step_reads(tmp_path, monkeypatch):
    entry = import_in("dataset_loading", "app.entry")
    monkeypatch.setattr(entry, "DATASET_DIR", str(tmp_path))
    monkeypatch.setattr(entry, "OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "data.yaml").write_text("kpt_shape: [4, 3]\n")
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(
        {"bucket": "dataset-cats", "prefix": "c0ffee/", "label_keys": {"train": ["x"]}, "lakefs_commit": "c0ffee"}))
    entry._write_outputs({"dataset": "dataset-cats@v2"})
    summary = json.loads((tmp_path / "out" / "manifest-summary.json").read_text())
    assert summary == {"bucket": "dataset-cats", "prefix": "c0ffee/", "label_keys": {"_present": True},
                       "ref": "dataset-cats@v2", "repo": "dataset-cats", "version": "v2",
                       "dataset_commit": "c0ffee"}


class _S3:
    """head_object over a fixed key set (the lakeFS S3 gateway: bucket = repository)."""

    def __init__(self, keys):
        self.keys = set(keys)

    def head_object(self, Bucket, Key):
        if f"{Bucket}/{Key}" not in self.keys:
            raise KeyError(Key)
        return {}


def _loading_service():
    mod = import_in("dataset_loading", "app.services.dataset_loading")
    svc = mod.DatasetLoadingService.__new__(mod.DatasetLoadingService)
    svc._logger = mod.logging.getLogger("test")
    return mod, svc


def _resolve_source(keys, commit="c0ffee"):
    with step_app("dataset_loading"):
        models = import_in("dataset_loading", "app.models.dataset")
        _, svc = _loading_service()
        svc._s3 = _S3(keys)
        params = models.YoloDatasetParams(version="dataset-cats@latest", source="lakefs", output_dir="/tmp/x",
                                          lakefs_repo="dataset-cats", lakefs_ref=commit)
        return svc._resolve_source(params, commit)


def test_should_read_the_pinned_commit_at_the_repository_root():
    assert _resolve_source({"dataset-cats/c0ffee/data.yaml"}) == ("dataset-cats", "c0ffee/")


@pytest.mark.parametrize("keys", [
    set(),
    {"dataset-cats/c0ffee/dataset/data.yaml"},         # the old {ref}/dataset/ layout is not read any more
    {"dataset-cats/c0ffee/dataset/cats/data.yaml"},
    {"dataset-cats/c0ffee/images/train/a.jpg"},         # no data.yaml
])
def test_should_have_no_legacy_layout_fallback(keys):
    with pytest.raises(Exception, match="no data.yaml at s3://dataset-cats/c0ffee/") as exc:
        _resolve_source(keys)
    assert type(exc.value).__name__ == "DatasetLoadingError"


def test_dataset_loading_has_no_lakefs_commit_lookup_of_its_own():
    src = (ROOT / "steps/dataset_loading/app/services/dataset_loading.py").read_text()
    assert "_fetch_lakefs_commit" not in src and "import requests" not in src


# ---------------------------------------------------------------- config-validation

def _validation():
    mod = import_in("config_validation", "app.services.config_validation")
    return mod, mod.ConfigValidationService()


def test_should_accept_a_dataset_version_with_data_yaml(client):
    mod, svc = _validation()
    from kaos_lib import datasets
    svc._check_dataset(datasets.DatasetRef.parse("dataset-cats@v1"), client)
    svc._check_dataset(datasets.DatasetRef.parse("dataset-cats@latest"), client)


def test_should_say_which_versions_exist(client):
    mod, svc = _validation()
    from kaos_lib import datasets
    with pytest.raises(mod.ConfigValidationError, match="no version v9.*Available: v1"):
        svc._check_dataset(datasets.DatasetRef.parse("dataset-cats@v9"), client)
    with pytest.raises(mod.ConfigValidationError, match="does not exist"):
        svc._check_dataset(datasets.DatasetRef.parse("dataset-dogs@v1"), client)


def test_should_refuse_a_dataset_without_data_yaml(fake, client):
    mod, svc = _validation()
    from kaos_lib import datasets
    fake.add_repo("dataset-empty", files={"README.md": b"x"})
    fake.tag("dataset-empty", "v1")
    with pytest.raises(mod.ConfigValidationError, match="has no data.yaml"):
        svc._check_dataset(datasets.DatasetRef.parse("dataset-empty@v1"), client)


@pytest.mark.parametrize("value", MALFORMED)
def test_should_reject_a_malformed_dataset_even_without_credentials(monkeypatch, value):
    mod, svc = _validation()
    for var in ("LAKEFS_ENDPOINT", "MLFLOW_TRACKING_URI"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(mod.ConfigValidationError):
        svc.run({"dataset": value, "data": {}, "model": {}, "platform": {}})


# ---------------------------------------------------------------- one pinned read for qat / quantization / MeluXina

@pytest.mark.parametrize("step", ["model_training", "qat_finetune", "model_quantization"])
def test_should_read_the_pinned_commit_even_when_latest_moves(step, fake, client, tmp_path):
    lio = import_in(step, "app.services.lakefs_io")
    pinned = fake.head("dataset-cats")
    r = fake.repos["dataset-cats"]
    moved = dict(r["commits"][pinned])
    moved["data.yaml"] = b"path: .\ntrain: images/train\nval: images/val\nnc: 2\nnames:\n  0: a\n  1: b\n"
    r["commits"]["c-new"] = moved
    r["branches"]["main"]["commit"] = "c-new"
    data_yaml = lio.materialize("dataset-cats@latest", pinned, str(tmp_path / "ds"), client)
    loaded = yaml.safe_load(data_yaml.read_text())
    assert loaded["nc"] == 1, "still the commit training pinned"
    assert loaded["path"] == str((tmp_path / "ds").resolve()), "path: points at the local copy"
    assert (tmp_path / "ds" / "images/train/a.jpg").read_bytes() == b"img"
    assert not (tmp_path / "ds" / "README.md").exists(), "only data.yaml, images/, labels/"


@pytest.mark.parametrize("step", ["model_training", "qat_finetune", "model_quantization"])
def test_should_stop_when_nothing_was_pinned(step, client, tmp_path):
    lio = import_in(step, "app.services.lakefs_io")
    with pytest.raises(SystemExit, match="no pinned dataset"):
        lio.materialize("dataset-cats@latest", "", str(tmp_path), client)
    with pytest.raises(SystemExit, match="no pinned dataset"):
        lio.materialize("", "c0ffee", str(tmp_path), client)


def test_should_use_the_copy_staged_on_meluxina_without_downloading(fake, client, tmp_path, monkeypatch):
    lio = import_in("model_training", "app.services.lakefs_io")
    commit = fake.head("dataset-cats")
    staged = tmp_path / "dataset"                       # bind-mounted read-only on the node
    (staged / "images/train").mkdir(parents=True)
    (staged / "data.yaml").write_bytes(DATA_YAML)
    (staged / ".kaos-ready").write_text(json.dumps({"repo": "dataset-cats", "commit": commit}))
    monkeypatch.setenv("KUBECORE_STAGED_DIR", str(staged))
    fake.calls.clear()
    data_yaml = lio.materialize("dataset-cats@v1", commit, str(tmp_path / "work"), client)
    assert yaml.safe_load(data_yaml.read_text())["path"] == str(staged)
    assert not [c for c in fake.calls if c[1].endswith("/objects")], "nothing downloaded"


def test_training_uses_the_consumer_only_when_a_copy_is_staged(monkeypatch):
    entry = import_in("model_training", "app.entry")
    assert entry._staged() is False
    monkeypatch.setenv("KUBECORE_STAGED_DIR", "/kubecore/dataset")
    assert entry._staged() is True


def test_training_location_is_the_pinned_manifest_summary():
    entry = import_in("model_training", "app.entry")
    summary = {"bucket": "dataset-cats", "prefix": "c0ffee/", "dataset_commit": "c0ffee", "ref": "dataset-cats@v2"}
    assert entry._dataset_location(summary) == ("dataset-cats", "c0ffee/", "c0ffee")
    with pytest.raises(SystemExit, match="no pinned dataset commit"):  # never guess a location
        entry._dataset_location({})


# ---------------------------------------------------------------- results: outputs-<app>, main, one commit

@pytest.mark.parametrize("step", ["model_training", "qat_finetune", "model_quantization"])
def test_should_write_results_as_one_commit_on_main_of_the_outputs_repository(step, fake, client, tmp_path, monkeypatch):
    lio = import_in(step, "app.services.lakefs_io")
    monkeypatch.setenv("KUBECORE_RUN_ID", "yolo-pipeline-x7k2p")
    f = tmp_path / "model_int8.tflite"
    f.write_bytes(b"tflite")
    prefix = lio.write_outputs("outputs-yolo", "exp", [f], kind="quantization", client=client,
                               metadata={"dataset_ref": "dataset-cats@v1"})
    assert prefix == "s3://outputs-yolo/main/quantization/exp/yolo-pipeline-x7k2p/"
    tree = fake.tree("outputs-yolo", fake.head("outputs-yolo"))
    assert set(tree) == {"quantization/exp/yolo-pipeline-x7k2p/model_int8.tflite"}
    assert [e for e in fake.repos["outputs-yolo"]["commit_log"] if e[0] == "main"], "committed on main"


def test_should_read_a_checkpoint_back_from_the_s3_uri_we_return(fake, client, tmp_path, monkeypatch):
    lio = import_in("model_training", "app.services.lakefs_io")
    monkeypatch.setenv("KUBECORE_RUN_ID", "run-1")
    f = tmp_path / "best.pt"
    f.write_bytes(b"PT")
    prefix = lio.write_outputs("outputs-yolo", "exp", [f], client=client)
    out = lio.download_object(prefix + "best.pt", str(tmp_path / "dl"), client)
    assert Path(out).read_bytes() == b"PT" and prefix == "s3://outputs-yolo/main/checkpoints/exp/run-1/"


def _training_service(mod, tmp_path, client, monkeypatch):
    """A TrainingService with an OPEN run writer on outputs-yolo (what _run_training sets up)."""
    from kaos_lib import outputs
    monkeypatch.setenv("KUBECORE_RUN_ID", "run-1")
    svc = mod.TrainingService.__new__(mod.TrainingService)
    svc._logger = mod.logging.getLogger("test")
    svc._outputs_stage = tmp_path / "stage"
    svc._outputs_stage.mkdir()
    svc._run_writer = outputs.run("outputs-yolo", "exp", "run-1", client=client).open()
    svc._get_mlflow_run_id = lambda: "mlflow-1"
    return svc


def _params():
    return type("P", (), {"checkpoint_bucket": "outputs-yolo", "experiment_name": "exp",
                          "dataset_repo": "dataset-cats", "dataset_version": "v2",
                          "lakefs_commit": "c0ffee"})()


def _tree(fake, ref="main"):
    return fake.tree("outputs-yolo", fake.head("outputs-yolo", ref))


def test_training_saves_every_checkpoint_on_the_run_branch_not_on_main(fake, client, tmp_path, monkeypatch):
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    (tmp_path / "w.pt").write_bytes(b"w10")
    main_before = fake.head("outputs-yolo")
    svc._upload_to_s3(tmp_path / "w.pt", "outputs-yolo", "main/checkpoints/exp/run-1/epoch_0010.pt")
    assert fake.head("outputs-yolo") == main_before, "main is untouched until finish"
    tree = _tree(fake, "run-run-1-checkpoints")
    assert set(tree) == {"checkpoints/exp/run-1/epoch_0010.pt", "checkpoints/exp/run-1/last.pt"}


def test_training_finish_merges_one_commit_with_provenance_and_deletes_the_branch(fake, client, tmp_path, monkeypatch):
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    for name in ("best.pt", "last.pt", "epoch_0010.pt"):
        (tmp_path / name).write_bytes(name.encode())
        svc._upload_to_s3(tmp_path / name, "outputs-yolo", f"main/checkpoints/exp/run-1/{name}")
    svc._commit_outputs(_params())
    assert set(_tree(fake)) == {f"checkpoints/exp/run-1/{n}" for n in ("best.pt", "last.pt", "epoch_0010.pt")}
    merges = [e for e in fake.repos["outputs-yolo"]["commit_log"] if e[0] == "main"]
    assert len(merges) == 1, "one commit on main per step per run"
    meta = merges[0][-1] if isinstance(merges[0][-1], dict) else {}
    assert "run-run-1-checkpoints" not in fake.repos["outputs-yolo"]["branches"]
    assert meta.get("dataset_ref") == "dataset-cats@v2" and meta.get("dataset_commit") == "c0ffee"
    assert meta.get("mlflow_run_id") == "mlflow-1" and meta.get("run") == "run-1"


def test_a_retried_step_resumes_from_the_last_checkpoint_of_the_run_branch(fake, client, connected, tmp_path, monkeypatch):
    mod = import_in("model_training", "app.services.model_training")
    (tmp_path / "a").mkdir()
    first = _training_service(mod, tmp_path / "a", client, monkeypatch)
    (tmp_path / "w.pt").write_bytes(b"epoch-20-state")
    first._upload_to_s3(tmp_path / "w.pt", "outputs-yolo", "main/checkpoints/exp/run-1/epoch_0020.pt")
    # the pod is lost here: nothing merged, the branch stays. The retry:
    (tmp_path / "b").mkdir()
    retry = mod.TrainingService.__new__(mod.TrainingService)
    retry._logger = mod.logging.getLogger("test")
    resume = retry._open_run_writer(_params(), tmp_path / "b")
    assert resume is not None and resume.name == "last.pt" and resume.read_bytes() == b"epoch-20-state"


def test_a_first_attempt_has_nothing_to_resume(fake, client, connected, tmp_path, monkeypatch):
    mod = import_in("model_training", "app.services.model_training")
    monkeypatch.setenv("KUBECORE_RUN_ID", "fresh")
    svc = mod.TrainingService.__new__(mod.TrainingService)
    svc._logger = mod.logging.getLogger("test")
    assert svc._open_run_writer(_params(), tmp_path) is None


def test_training_fails_loudly_when_the_results_cannot_be_merged(fake, client, tmp_path, monkeypatch):
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    (tmp_path / "best.pt").write_bytes(b"x")
    svc._upload_to_s3(tmp_path / "best.pt", "outputs-yolo", "main/checkpoints/exp/run-1/best.pt")
    svc._run_writer.client.merge = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(mod.TrainingError, match="outputs-yolo"):
        svc._commit_outputs(_params())


def test_checkpoints_are_scoped_to_the_run(monkeypatch):
    mt = import_in("model_training", "app.services.model_training")
    params = type("P", (), {"checkpoint_prefix": "main/checkpoints/", "experiment_name": "exp"})()
    monkeypatch.setenv("KUBECORE_RUN_ID", "yolo-pipeline-x7k2p")
    assert mt.checkpoint_base(params) == "main/checkpoints/exp/yolo-pipeline-x7k2p"
    monkeypatch.delenv("KUBECORE_RUN_ID")
    assert mt.checkpoint_base(params) == "main/checkpoints/exp"


@pytest.mark.parametrize("step", ["qat_finetune", "model_quantization"])
def test_should_have_no_s3_client_or_lakefs_copy_left(step):
    """The INT8 artifact goes through write_run; checkpoints/tflite come back through the package client."""
    app = ROOT / "steps" / step / "app"
    assert not (app / "services" / "lakefs_objects.py").exists()
    svc = (app / "services" / ("qat_service.py" if step == "qat_finetune" else "quantization_service.py")).read_text()
    assert "lakefs_io.write_outputs(" in svc and 'kind="quantization"' in svc
    assert "lakefs_io.download_object(" in svc
    assert "self._s3" not in svc and "boto3" not in (app / "manager.py").read_text()


@pytest.mark.parametrize("step", ["qat_finetune", "model_quantization"])
def test_calibration_dataset_follows_training(step):
    """The step reads the snapshot training read (training-result), via the consumer."""
    src = (ROOT / "steps" / step / "app" / "entry.py").read_text()
    assert 'lakefs_io.materialize(_ref, str(tr.get("dataset_commit") or "")' in src
    assert "_download_prefix" not in src and "_staged_dataset_dir" not in src


def test_the_three_lakefs_io_files_are_identical():
    """Each step image carries only its own app/, so the file is copied, never imported across steps."""
    texts = {s: (ROOT / "steps" / s / "app/services/lakefs_io.py").read_text()
             for s in ("model_training", "qat_finetune", "model_quantization")}
    assert len(set(texts.values())) == 1


def test_training_provenance_logs_dataset_repo_version_and_commit():
    src = (ROOT / "steps/model_training/app/services/model_training.py").read_text()
    assert '"dataset.repo": params.dataset_repo' in src
    assert '"dataset.version": params.dataset_version' in src
    assert '"dataset.lakefs_commit": params.lakefs_commit' in src


def test_step_images_install_kaos_lib_and_not_the_vendored_engine():
    specs = set()
    for step in ("dataset_loading", "config_validation", "model_training", "qat_finetune", "model_quantization",
                 "model_registration"):
        text = (ROOT / "steps" / step / "Dockerfile").read_text()
        assert 'pip install --no-cache-dir "kaos-lib[zitadel] ${KAOS_LIB_PACKAGE}"' in text, step
        assert "COPY kubecore" not in text and "hera-pipeline-template" not in text, step
        specs.add(next(l.split("=", 1)[1] for l in text.splitlines() if l.startswith("ARG KAOS_LIB_PACKAGE=")))
    assert len(specs) == 1, "one kaos-lib pin for every step"
    spec = next(iter(specs))
    assert re.fullmatch(r'"@ https://github\.com/kaos-io/kaos-lib/archive/[0-9a-f]{40}\.zip"', spec), spec


def test_should_pin_a_branch_pick_to_its_commit_too(client, fake):
    entry = import_in("dataset_loading", "app.entry")
    p = entry._resolve_dataset_params({"dataset": "dataset-cats@main", "data": {"source": "lakefs"}}, client)
    assert p["lakefs_ref"] == fake.head("dataset-cats"), "a moving branch is resolved once, to a commit"
