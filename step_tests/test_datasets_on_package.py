"""PRD novelcore/kaos#1283 F-07 / F-14: the steps read datasets and write results through kaos-lib (kaos_lib).

A run picks `<dataset-repo>@<version>`; the platform pins it to ONE commit (the data lock); training builds its
dataset location from that commit, and training, qat and quantization read it (consumer) and write to
`outputs-<app>` (write_run).
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


# ---------------------------------------------------------------- training builds its dataset location

def _lock(commit, repo="dataset-cats", ref="latest"):
    return json.dumps({"version": 1, "reads": [{"repo": repo, "ref": ref, "commit": commit}]})


def _move_latest(fake):
    """Someone merges a new dataset version: `latest` now points at a commit with another data.yaml."""
    r = fake.repos["dataset-cats"]
    old = r["branches"]["main"]["commit"]
    moved = dict(r["commits"][old])
    moved["data.yaml"] = b"path: .\ntrain: images/train\nval: images/val\nnc: 2\nnames:\n  0: a\n  1: b\n"
    r["commits"][NEW] = moved
    r["branches"]["main"]["commit"] = NEW
    return old


NEW = "b" * 40


def test_training_builds_its_dataset_location_from_the_lock_even_when_latest_moves(fake, client, monkeypatch):
    entry = import_in("model_training", "app.entry")
    locked = _move_latest(fake)
    monkeypatch.setenv("KAOS_LIB_LOCK", _lock(locked))
    loc = entry._dataset_location({"dataset": "dataset-cats@latest"}, client)
    assert loc == {"bucket": "dataset-cats", "prefix": f"{locked}/", "dataset_commit": locked,
                   "ref": "dataset-cats@latest", "repo": "dataset-cats", "version": "latest", "shared": False}
    assert entry._dataset_yaml("dataset-cats", locked, client) == DATA_YAML.decode(), "data.yaml of the locked commit"


def test_training_reads_the_head_of_the_pick_when_the_run_has_no_lock(fake, client):
    entry = import_in("model_training", "app.entry")
    first = _move_latest(fake)
    loc = entry._dataset_location({"dataset": "dataset-cats@latest"}, client)
    assert loc["dataset_commit"] == NEW and loc["prefix"] == NEW + "/"
    assert "nc: 2" in entry._dataset_yaml("dataset-cats", NEW, client)
    loc = entry._dataset_location({"dataset": "dataset-cats@v1"}, client)
    assert loc["dataset_commit"] == first, "v1 is a tag on the first commit"


def test_training_stops_when_the_version_does_not_exist(client):
    entry = import_in("model_training", "app.entry")
    with pytest.raises(DatasetError, match="no version v7.*Available: v1"):
        entry._dataset_location({"dataset": "dataset-cats@v7"}, client)
    with pytest.raises(DatasetError, match="does not exist"):
        entry._dataset_location({"dataset": "dataset-dogs@v1"}, client)


@pytest.mark.parametrize("value", MALFORMED)
def test_training_refuses_a_malformed_dataset_with_a_clear_message(client, value):
    entry = import_in("model_training", "app.entry")
    with pytest.raises(DatasetError):
        entry._dataset_location({"dataset": value}, client)


def test_training_refuses_an_s3_path_override_instead_of_ignoring_it(client):
    entry = import_in("model_training", "app.entry")
    with pytest.raises(SystemExit, match="path_override is not read any more"):
        entry._dataset_location({"dataset": "dataset-cats@v1", "data": {"path_override": "s3://b/p/"}}, client)
    entry._dataset_location({"dataset": "dataset-cats@v1", "data": {"path_override": ""}}, client)


def test_training_stops_on_a_dataset_without_data_yaml(fake, client):
    entry = import_in("model_training", "app.entry")
    fake.add_repo("dataset-empty", files={"README.md": b"x"})
    with pytest.raises(SystemExit, match="has no data.yaml"):
        entry._dataset_yaml("dataset-empty", fake.head("dataset-empty"), client)


def test_training_marks_a_shared_dataset(monkeypatch):
    """A dataset another project shares: training materializes it through kaos-lib instead of streaming bucket/prefix."""
    entry = import_in("model_training", "app.entry")
    _, owner, owner_client = _owner_world(monkeypatch)
    loc = entry._dataset_location({"dataset": "acme/dataset-x@v1"})
    assert loc["shared"] is True and loc["ref"] == "acme/dataset-x@v1"
    assert loc["bucket"] == "dataset-x" and loc["dataset_commit"] == owner.head("dataset-x")


def test_no_step_reads_what_dataset_loading_used_to_hand_over():
    for step in ("model_training", "qat_finetune", "model_quantization", "model_registration"):
        src = (ROOT / "steps" / step / "app" / "entry.py").read_text()
        assert "--manifest-summary" not in src and "--data-yaml" not in src and "--dataset-info" not in src, step


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
    monkeypatch.setenv("KUBECORE_STAGED_DIR", str(staged))
    work = tmp_path / "work"
    data_yaml = lio.materialize("dataset-cats@v1", commit, str(work), client)
    assert data_yaml.parent == work and not (work / "images").exists()
    assert lio.dataset_root(data_yaml) == str(staged)
    # a downloaded copy keeps data.yaml and the data together
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
    monkeypatch.setenv("KUBECORE_STAGED_DIR", "/kubecore/dataset")
    assert entry._staged() is True


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


def test_training_saves_a_result_on_the_run_branch_not_on_main(fake, client, tmp_path, monkeypatch):
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    (tmp_path / "w.pt").write_bytes(b"w10")
    main_before = fake.head("outputs-yolo")
    svc._upload_to_s3(tmp_path / "w.pt", "outputs-yolo", "main/checkpoints/exp/run-1/best.pt")
    assert fake.head("outputs-yolo") == main_before, "main is untouched until finish"
    tree = _tree(fake, "run-run-1-checkpoints")
    assert set(tree) == {"checkpoints/exp/run-1/best.pt"}


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


def test_a_retried_step_does_not_resume_from_the_run_branch(fake, client, connected, tmp_path, monkeypatch):
    """One resume mechanism: the platform checkpoint folder. A saved result on the run branch is not a resume point."""
    mod = import_in("model_training", "app.services.model_training")
    (tmp_path / "a").mkdir()
    first = _training_service(mod, tmp_path / "a", client, monkeypatch)
    (tmp_path / "w.pt").write_bytes(b"state")
    first._upload_to_s3(tmp_path / "w.pt", "outputs-yolo", "main/checkpoints/exp/run-1/last.pt")
    retry = mod.TrainingService.__new__(mod.TrainingService)
    retry._logger = mod.logging.getLogger("test")
    assert retry._open_run_writer(_params()) is None
    assert retry._run_writer is not None


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
    for step in ("config_validation", "model_training", "qat_finetune", "model_quantization",
                 "model_registration"):
        text = (ROOT / "steps" / step / "Dockerfile").read_text()
        assert 'pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"' in text, step   # the WHOLE requirement
        assert "COPY kubecore" not in text and "hera-pipeline-template" not in text, step
        specs.add(next(l.split("=", 1)[1] for l in text.splitlines() if l.startswith("ARG KAOS_LIB_PACKAGE=")))
    assert len(specs) == 1, "one kaos-lib pin for every step"
    spec = next(iter(specs))
    assert re.fullmatch(r'"kaos-lib\[zitadel\] @ https://github\.com/kaos-io/kaos-lib/archive/[0-9a-f]{40}\.zip"', spec), spec


# ---------------------------------------------------------------- shared datasets: training

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


def test_training_reads_a_shared_dataset_through_kaos_lib_not_s3(monkeypatch, tmp_path):
    """The shared dataset exists only on the owner: training must materialize it, not stream bucket/prefix."""
    _, owner, oc = _owner_world(monkeypatch)
    lio = import_in("model_training", "app.services.lakefs_io")
    data_yaml = lio.materialize("acme/dataset-x@v1", owner.head("dataset-x"), str(tmp_path / "ds"), oc)
    assert (tmp_path / "ds" / "images/train/a.jpg").read_bytes() == b"img"
    assert yaml.safe_load(data_yaml.read_text())["path"] == str((tmp_path / "ds").resolve())
