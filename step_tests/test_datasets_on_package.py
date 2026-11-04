"""PRD novelcore/kaos#1283 F-07 / F-14: the steps read datasets and write results through kaos-lib (kaos_lib).

A run picks `<dataset-repo>@<version>`; dataset-loading resolves it to ONE commit; training, qat and
quantization read that commit (consumer) and write to `outputs-<app>` (write_run).
"""
import importlib
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


def _shared_world(monkeypatch, owner_has_yaml=True):
    """A consumer lakeFS WITHOUT the dataset and an owner lakeFS WITH it; `datasets.source` routes a
    ref carrying an owner to the owner's client, as kaos-lib does through the peers."""
    from fake_lakefs import FakeLakeFS
    from kaos_lib import datasets
    consumer, owner = FakeLakeFS(), FakeLakeFS()
    owner.add_repo("dataset-x", files={"data.yaml": DATA_YAML} if owner_has_yaml else {"README.md": b"x"})
    owner.tag("dataset-x", "v1")
    owner_client = owner.client()
    monkeypatch.setattr(datasets, "source", lambda ref, client=None: (
        (ref, owner_client) if ref.owner else (ref, client)))
    return consumer.client()


def test_should_read_a_shared_dataset_from_the_owners_lakefs(monkeypatch):
    mod, svc = _validation()
    from kaos_lib import datasets
    consumer = _shared_world(monkeypatch)
    svc._check_dataset(datasets.DatasetRef.parse("acme/dataset-x@v1"), consumer)


def test_should_refuse_a_shared_dataset_without_data_yaml_on_the_owners_lakefs(monkeypatch):
    mod, svc = _validation()
    from kaos_lib import datasets
    consumer = _shared_world(monkeypatch, owner_has_yaml=False)
    with pytest.raises(mod.ConfigValidationError, match="has no data.yaml"):
        svc._check_dataset(datasets.DatasetRef.parse("acme/dataset-x@v1"), consumer)


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
    monkeypatch.setenv("KAOS_STAGED_DIR", str(staged))
    monkeypatch.setenv("KUBECORE_STAGED_DIR", str(staged))  # deprecated alias, an older kaos-lib reads it
    fake.calls.clear()
    data_yaml = lio.materialize("dataset-cats@v1", commit, str(tmp_path / "work"), client)
    assert yaml.safe_load(data_yaml.read_text())["path"] == str(staged)
    assert not [c for c in fake.calls if c[1].endswith("/objects")], "nothing downloaded"


@pytest.mark.parametrize("step", ["model_training", "qat_finetune", "model_quantization"])
def test_should_hand_the_trainer_the_folder_that_holds_the_images_not_the_data_yaml_folder(
        step, fake, client, tmp_path, monkeypatch):
    """HPC (e2e ml_hpc, jobs 5418743/5420267): the staged copy is used in place, so the data.yaml
    lives in the workspace while images/ and labels/ live in the staged folder. dataset_dir must be
    the yaml's path:, or the trainer stops with 'images/train/ directory not found'."""
    lio = import_in(step, "app.services.lakefs_io")
    commit = fake.head("dataset-cats")
    staged = tmp_path / "staged"
    (staged / "images/train").mkdir(parents=True)
    (staged / "data.yaml").write_bytes(DATA_YAML)
    (staged / ".kaos-ready").write_text(json.dumps({"repo": "dataset-cats", "commit": commit}))
    monkeypatch.setenv("KAOS_STAGED_DIR", str(staged))
    monkeypatch.setenv("KUBECORE_STAGED_DIR", str(staged))  # deprecated alias, an older kaos-lib reads it
    work = tmp_path / "work"
    data_yaml = lio.materialize("dataset-cats@v1", commit, str(work), client)
    assert data_yaml.parent == work and not (work / "images").exists()
    assert lio.dataset_root(data_yaml) == str(staged)
    # a downloaded copy keeps data.yaml and the data together
    monkeypatch.delenv("KAOS_STAGED_DIR")
    monkeypatch.delenv("KUBECORE_STAGED_DIR")
    data_yaml = lio.materialize("dataset-cats@v1", commit, str(tmp_path / "dl"), client)
    assert lio.dataset_root(data_yaml) == str((tmp_path / "dl").resolve())


@pytest.mark.parametrize("step", ["model_training", "qat_finetune", "model_quantization"])
def test_steps_pass_the_dataset_root_to_the_trainer(step):
    src = (ROOT / "steps" / step / "app" / "entry.py").read_text()
    assert "lakefs_io.dataset_root(data_yaml)" in src and "data_yaml.parent" not in src


def test_training_uses_the_consumer_only_when_a_copy_is_staged(monkeypatch):
    entry = import_in("model_training", "app.entry")
    assert entry._staged() is False
    monkeypatch.setenv("KAOS_STAGED_DIR", "/kaos/dataset")
    monkeypatch.setenv("KUBECORE_STAGED_DIR", "/kaos/dataset")  # deprecated alias, an older kaos-lib reads it
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
    monkeypatch.setenv("KAOS_RUN_ID", "yolo-pipeline-x7k2p")
    monkeypatch.setenv("KUBECORE_RUN_ID", "yolo-pipeline-x7k2p")  # deprecated alias, an older kaos-lib reads it
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
    monkeypatch.setenv("KAOS_RUN_ID", "run-1")
    monkeypatch.setenv("KUBECORE_RUN_ID", "run-1")  # deprecated alias, an older kaos-lib reads it
    f = tmp_path / "best.pt"
    f.write_bytes(b"PT")
    prefix = lio.write_outputs("outputs-yolo", "exp", [f], client=client)
    out = lio.download_object(prefix + "best.pt", str(tmp_path / "dl"), client)
    assert Path(out).read_bytes() == b"PT" and prefix == "s3://outputs-yolo/main/checkpoints/exp/run-1/"


def _training_service(mod, tmp_path, client, monkeypatch):
    """A TrainingService with an OPEN run writer on outputs-yolo (what _run_training sets up)."""
    from kaos_lib import outputs
    monkeypatch.setenv("KAOS_RUN_ID", "run-1")
    monkeypatch.setenv("KUBECORE_RUN_ID", "run-1")  # deprecated alias, an older kaos-lib reads it
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
    monkeypatch.setenv("KAOS_RUN_ID", "fresh")
    monkeypatch.setenv("KUBECORE_RUN_ID", "fresh")  # deprecated alias, an older kaos-lib reads it
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
    monkeypatch.setenv("KAOS_RUN_ID", "yolo-pipeline-x7k2p")
    monkeypatch.setenv("KUBECORE_RUN_ID", "yolo-pipeline-x7k2p")  # deprecated alias, an older kaos-lib reads it
    assert mt.checkpoint_base(params) == "main/checkpoints/exp/yolo-pipeline-x7k2p"
    monkeypatch.delenv("KAOS_RUN_ID")
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
        assert 'pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"' in text, step   # the WHOLE requirement
        assert "COPY kubecore" not in text and "COPY kaos_pipeline" not in text and "hera-pipeline-template" not in text, step
        specs.add(next(l.split("=", 1)[1] for l in text.splitlines() if l.startswith("ARG KAOS_LIB_PACKAGE=")))
    assert len(specs) == 1, "one kaos-lib pin for every step"
    spec = next(iter(specs))
    assert re.fullmatch(r'"kaos-lib\[zitadel\] @ https://github\.com/kaos-io/kaos-lib/archive/[0-9a-f]{40}\.zip"', spec), spec


def test_should_pin_a_branch_pick_to_its_commit_too(client, fake):
    entry = import_in("dataset_loading", "app.entry")
    p = entry._resolve_dataset_params({"dataset": "dataset-cats@main", "data": {"source": "lakefs"}}, client)
    assert p["lakefs_ref"] == fake.head("dataset-cats"), "a moving branch is resolved once, to a commit"


# ---------------------------------------------------------------- shared datasets: dataset-loading -> training

class _NoGateway:
    """The consumer's S3 gateway: the shared dataset is not there, and any call is a failure."""
    def __getattr__(self, name):
        raise AssertionError(f"consumer S3 gateway used: {name}")


POSE_YAML = b"path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: ship\nkpt_shape: [1, 3]\n"
POSE_LABEL = b"0 0.5 0.5 0.1 0.1 0.5 0.5 2\n"


def _owner_world(monkeypatch):
    """The dataset exists ONLY on the owner's lakeFS; `datasets.source` routes an owned ref there."""
    from fake_lakefs import FakeLakeFS
    from kaos_lib import datasets
    consumer, owner = FakeLakeFS(), FakeLakeFS()
    owner.add_repo("dataset-x", files={"data.yaml": POSE_YAML, "images/train/a.jpg": b"img",
                                       "labels/train/a.txt": POSE_LABEL,
                                       "images/val/b.jpg": b"img", "labels/val/b.txt": POSE_LABEL})
    owner.tag("dataset-x", "v1")
    owner_client = owner.client()
    monkeypatch.setattr(datasets, "source", lambda ref, client=None: (
        (ref, owner_client) if ref.owner else (ref, client or consumer.client())))
    return consumer, owner, owner_client


def test_should_serve_list_head_and_download_from_the_owners_lakefs(monkeypatch, tmp_path):
    _, owner, oc = _owner_world(monkeypatch)
    objects = import_in("dataset_loading", "app.services.owner_objects").OwnerObjects(oc)
    commit = owner.head("dataset-x")
    keys = [o["Key"] for page in objects.get_paginator("list_objects_v2").paginate(
        Bucket="dataset-x", Prefix=f"{commit}/") for o in page["Contents"]]
    assert sorted(keys) == [f"{commit}/{p}" for p in (
        "data.yaml", "images/train/a.jpg", "images/val/b.jpg", "labels/train/a.txt", "labels/val/b.txt")]
    objects.head_object(Bucket="dataset-x", Key=f"{commit}/data.yaml")
    with pytest.raises(Exception):
        objects.head_object(Bucket="dataset-x", Key=f"{commit}/missing.yaml")
    objects.download_file("dataset-x", f"{commit}/labels/train/a.txt", str(tmp_path / "x" / "a.txt"))
    assert (tmp_path / "x" / "a.txt").read_bytes() == POSE_LABEL


def test_should_load_a_shared_dataset_without_the_consumers_s3_gateway(monkeypatch, tmp_path):
    _, owner, oc = _owner_world(monkeypatch)
    entry = import_in("dataset_loading", "app.entry")
    with step_app("dataset_loading"):
        svc_mod = importlib.import_module("app.services.dataset_loading")
        models = importlib.import_module("app.models.dataset")
    cfg = {"dataset": "acme/dataset-x@v1", "data": {"source": "lakefs"}}
    objects = entry._owner_objects(cfg)
    assert objects is not None
    params = entry._resolve_dataset_params(cfg)
    assert params["lakefs_ref"] == owner.head("dataset-x")
    params["output_dir"] = str(tmp_path / "out")
    svc = svc_mod.DatasetLoadingService(s3_client=objects)
    svc.run(models.YoloDatasetParams(**params))
    assert yaml.safe_load((tmp_path / "out" / "data.yaml").read_text())["kpt_shape"] == [1, 3]
    manifest = json.loads((tmp_path / "out" / "dataset_manifest.json").read_text())
    assert manifest["bucket"] == "dataset-x" and manifest["prefix"] == owner.head("dataset-x") + "/"
    assert manifest["total_images"] == 2


def test_should_keep_the_s3_gateway_for_the_projects_own_dataset(fake, client):
    entry = import_in("dataset_loading", "app.entry")
    assert entry._owner_objects({"dataset": "dataset-cats@v1", "data": {"source": "lakefs"}}, client) is None
    # an s3 source or an explicit override never goes through lakeFS
    assert entry._owner_objects({"dataset": "acme/dataset-x@v1", "data": {"source": "s3"}}, client) is None
    assert entry._owner_objects({"dataset": "acme/dataset-x@v1",
                                 "data": {"source": "lakefs", "path_override": "s3://b/p/"}}, client) is None


def test_should_use_the_gateway_service_path_for_an_own_dataset(fake, client, tmp_path):
    """Regression: with no owner client the service still reads through the injected S3 client."""
    with step_app("dataset_loading"):
        svc_mod = importlib.import_module("app.services.dataset_loading")
    calls = []

    class Gateway:
        def head_object(self, Bucket, Key):
            calls.append(("head", Bucket, Key))
        def get_paginator(self, op):
            calls.append(("paginator", op))
            raise RuntimeError("stop after the gateway was used")
    svc = svc_mod.DatasetLoadingService(s3_client=Gateway())
    assert svc._key_exists("dataset-cats", "c1/data.yaml") is True
    assert calls == [("head", "dataset-cats", "c1/data.yaml")]


def test_dataset_loading_marks_a_shared_dataset_for_training(tmp_path, monkeypatch):
    entry = import_in("dataset_loading", "app.entry")
    monkeypatch.setattr(entry, "DATASET_DIR", str(tmp_path))
    monkeypatch.setattr(entry, "OUTPUT_DIR", tmp_path / "out")
    (tmp_path / "data.yaml").write_text("kpt_shape: [4, 3]\n")
    (tmp_path / "dataset_manifest.json").write_text(json.dumps(
        {"bucket": "dataset-x", "prefix": "c0ffee/", "lakefs_commit": "c0ffee", "label_keys": {"a": 1}}))
    entry._write_outputs({"dataset": "acme/dataset-x@v1"}, shared=True)
    assert json.loads((tmp_path / "out" / "manifest-summary.json").read_text())["shared"] is True
    entry._write_outputs({"dataset": "dataset-x@v1"})
    assert "shared" not in json.loads((tmp_path / "out" / "manifest-summary.json").read_text())


def test_training_reads_a_shared_dataset_through_kaos_lib_not_s3(monkeypatch, tmp_path):
    """The shared dataset exists only on the owner: training must materialize it, not stream bucket/prefix."""
    _, owner, oc = _owner_world(monkeypatch)
    lio = import_in("model_training", "app.services.lakefs_io")
    data_yaml = lio.materialize("acme/dataset-x@v1", owner.head("dataset-x"), str(tmp_path / "ds"), oc)
    assert (tmp_path / "ds" / "images/train/a.jpg").read_bytes() == b"img"
    assert yaml.safe_load(data_yaml.read_text())["path"] == str((tmp_path / "ds").resolve())


def test_the_run_id_is_read_under_either_name(monkeypatch):
    """KAOS_RUN_ID is the name; KUBECORE_RUN_ID is the deprecated alias an older engine still sets."""
    lakefs_io = import_in("model_training", "app.services.lakefs_io")
    monkeypatch.setenv("KUBECORE_RUN_ID", "wf-old")
    assert lakefs_io.run_id() == "wf-old"
    monkeypatch.setenv("KAOS_RUN_ID", "wf-new")
    assert lakefs_io.run_id() == "wf-new"
