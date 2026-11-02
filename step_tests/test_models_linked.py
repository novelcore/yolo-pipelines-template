"""kaos#1283: a trained model is linked both ways between lakeFS (`outputs-<app>`) and MLflow.

Training records where its results landed on the MLflow run (`kaos.outputs_*`, `kaos.workflow`)
and logs best.pt as an MLflow 3 logged model (the experiment's Models tab) carrying the same
tags; registration registers best.pt from that logged model, tags the commit in lakeFS
(`model-<slug(name)>-v<N>`) and records the tag on the model version. MLflow is a recording
fake; lakeFS is the package's in-memory fake.
"""
import logging
import re
import sys
import types

import pytest
from conftest import import_in


class FakeMlflow:
    """The slice of mlflow the steps call: register_model, get_run, run/version tags, aliases,
    logged models (create_external_model, search, tags) and metrics."""

    EXPERIMENT_ID = "7"

    def __init__(self, run_tags=None, fail_create=False):
        self.run_tags = {"mlflow-1": dict(run_tags or {})}
        self.versions = []            # (name, model_uri) in registration order: version = index + 1
        self.version_tags = {}        # (name, version) -> {key: value}
        self.logged_models = []       # create_external_model kwargs + model_id, in creation order
        self.metrics = []             # (run_id, key, value, step, model_id)
        self.fail_create = fail_create

    def module(self):
        fake = self

        class MlflowClient:
            def set_tag(self, run_id, key, value):
                fake.run_tags.setdefault(run_id, {})[key] = value

            def get_run(self, run_id):
                return types.SimpleNamespace(data=types.SimpleNamespace(tags=dict(fake.run_tags.get(run_id, {}))),
                                             info=types.SimpleNamespace(experiment_id=fake.EXPERIMENT_ID))

            def search_logged_models(self, experiment_ids, filter_string, max_results=None):
                want = dict(re.findall(r"(\w+) = '([^']*)'", filter_string))
                found = [types.SimpleNamespace(**m) for m in fake.logged_models
                         if m["experiment_id"] in experiment_ids
                         and all(m[k] == v for k, v in want.items())]
                return found[:max_results]

            def set_logged_model_tags(self, model_id, tags):
                next(m for m in fake.logged_models if m["model_id"] == model_id)["tags"].update(tags)

            def log_metric(self, run_id, key, value, step=None, model_id=None):
                fake.metrics.append((run_id, key, value, step, model_id))

            def set_model_version_tag(self, name, version, key, value):
                fake.version_tags.setdefault((name, int(version)), {})[key] = value

            def set_registered_model_alias(self, name, alias, version):
                pass

        def register_model(model_uri, name):
            fake.versions.append((name, model_uri))
            return types.SimpleNamespace(version=str(len(fake.versions)))

        def create_external_model(name, source_run_id, experiment_id, params, tags):
            if fake.fail_create:
                raise RuntimeError("403 Forbidden")
            model = {"model_id": f"m-{len(fake.logged_models) + 1}", "name": name,
                     "source_run_id": source_run_id, "experiment_id": experiment_id,
                     "params": dict(params), "tags": dict(tags)}
            fake.logged_models.append(model)
            return types.SimpleNamespace(**model)

        mod = types.ModuleType("mlflow")
        mod.set_tracking_uri = lambda uri: None
        mod.register_model = register_model
        mod.create_external_model = create_external_model
        mod.tracking = types.ModuleType("mlflow.tracking")
        mod.tracking.MlflowClient = MlflowClient
        return mod


def _install(monkeypatch, fake_mlflow):
    mod = fake_mlflow.module()
    monkeypatch.setitem(sys.modules, "mlflow", mod)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", mod.tracking)


def _merged_commit(fake):
    """A commit on main of outputs-yolo holding a run's checkpoints (what training's finish merged)."""
    branches = fake.repos["outputs-yolo"]["branches"]
    branches["main"]["staged"]["checkpoints/exp/run-1/best.pt"] = b"best"
    commit, _ = fake._new_commit("outputs-yolo", fake.tree("outputs-yolo", "main"))
    branches["main"] = {"commit": commit, "staged": {}}
    return commit


# ---------------------------------------------------------------- training -> MLflow run

def _training_service(mod, tmp_path, client, monkeypatch):
    from kaos_lib import outputs
    monkeypatch.setenv("KUBECORE_RUN_ID", "wf-abc12")
    svc = mod.TrainingService.__new__(mod.TrainingService)
    svc._logger = mod.logging.getLogger("test")
    svc._outputs_stage = tmp_path / "stage"
    svc._outputs_stage.mkdir()
    svc._run_writer = outputs.run("outputs-yolo", "exp", "wf-abc12", client=client).open()
    svc._get_mlflow_run_id = lambda: "mlflow-1"
    return svc


def _params():
    return type("P", (), {"checkpoint_bucket": "outputs-yolo", "experiment_name": "exp",
                          "dataset_repo": "dataset-cats", "dataset_version": "v2",
                          "lakefs_commit": "c0ffee", "epochs": 50, "image_size": 640,
                          "model_variant": "yolov8n-pose.pt"})()


def test_training_returns_the_merge_commit_on_main(fake, client, tmp_path, monkeypatch):
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    (tmp_path / "best.pt").write_bytes(b"best")
    svc._upload_to_s3(tmp_path / "best.pt", "outputs-yolo", "main/checkpoints/exp/wf-abc12/best.pt")
    assert svc._commit_outputs(_params()) == fake.head("outputs-yolo")


def test_training_tags_the_mlflow_run_with_where_its_results_are(fake, client, tmp_path, monkeypatch):
    mlf = FakeMlflow()
    _install(monkeypatch, mlf)
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    monkeypatch.setenv("KUBECORE_WORKFLOW_NAME", "wf-abc12")
    svc._tag_outputs("mlflow-1", "c0ffee")
    assert mlf.run_tags["mlflow-1"] == {
        "kaos.outputs_repo": "outputs-yolo",
        "kaos.outputs_commit": "c0ffee",
        "kaos.outputs_path": "checkpoints/exp/wf-abc12/",
        "kaos.workflow": "wf-abc12",
    }


def test_training_workflow_tag_falls_back_to_the_run_id(fake, client, tmp_path, monkeypatch):
    """An enhancer that injects only KUBECORE_RUN_ID (= workflow.name) still names the workflow."""
    mlf = FakeMlflow()
    _install(monkeypatch, mlf)
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    svc._tag_outputs("mlflow-1", "c0ffee")
    assert mlf.run_tags["mlflow-1"]["kaos.workflow"] == "wf-abc12"


def test_training_without_an_mlflow_run_sets_no_tags(fake, client, tmp_path, monkeypatch):
    mlf = FakeMlflow()
    _install(monkeypatch, mlf)
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    svc._tag_outputs("", "c0ffee")
    assert mlf.run_tags == {"mlflow-1": {}}


# ---------------------------------------------------------------- training -> logged model

EPOCH_METRICS = {"val/precision": 0.9, "val/recall": 0.8, "val/mAP50": 0.95, "val/mAP50_95": 0.7,
                 "train/box_loss": 1.2}


def _log_model(monkeypatch, fake, client, tmp_path, mlf):
    _install(monkeypatch, mlf)
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    monkeypatch.setenv("KUBECORE_WORKFLOW_NAME", "wf-abc12")
    svc._log_model("mlflow-1", "c0ffee", _params(), EPOCH_METRICS, 50)
    return svc


def test_training_logs_best_as_a_model_of_the_run_with_its_settings_and_lakefs_tags(
        fake, client, tmp_path, monkeypatch):
    mlf = FakeMlflow()
    _log_model(monkeypatch, fake, client, tmp_path, mlf)
    assert mlf.logged_models == [{
        "model_id": "m-1", "name": "exp-best", "source_run_id": "mlflow-1", "experiment_id": "7",
        "params": {"epochs": "50", "imgsz": "640", "model_variant": "yolov8n-pose.pt"},
        "tags": {"kaos.outputs_repo": "outputs-yolo", "kaos.outputs_commit": "c0ffee",
                 "kaos.outputs_path": "checkpoints/exp/wf-abc12/", "kaos.workflow": "wf-abc12",
                 "kaos.dataset_ref": "dataset-cats@v2"}}]
    assert mlf.run_tags["mlflow-1"]["kaos.logged_model_id"] == "m-1"


def test_training_logs_the_final_val_metrics_against_the_model_at_the_last_epoch(
        fake, client, tmp_path, monkeypatch):
    mlf = FakeMlflow()
    _log_model(monkeypatch, fake, client, tmp_path, mlf)
    assert sorted(mlf.metrics) == sorted(
        ("mlflow-1", k, v, 50, "m-1") for k, v in EPOCH_METRICS.items() if k.startswith("val/"))


def test_training_retry_reuses_the_runs_logged_model(fake, client, tmp_path, monkeypatch):
    mlf = FakeMlflow()
    svc = _log_model(monkeypatch, fake, client, tmp_path, mlf)
    svc._log_model("mlflow-1", "beef", _params(), EPOCH_METRICS, 50)       # the retried step
    assert [m["model_id"] for m in mlf.logged_models] == ["m-1"], "searched before creating"
    assert mlf.logged_models[0]["tags"]["kaos.outputs_commit"] == "beef", "tags follow the retry"
    assert mlf.run_tags["mlflow-1"]["kaos.logged_model_id"] == "m-1"


def test_training_another_run_gets_its_own_logged_model(fake, client, tmp_path, monkeypatch):
    mlf = FakeMlflow()
    svc = _log_model(monkeypatch, fake, client, tmp_path, mlf)
    svc._log_model("mlflow-2", "beef", _params(), EPOCH_METRICS, 50)
    assert [(m["model_id"], m["source_run_id"]) for m in mlf.logged_models] == [
        ("m-1", "mlflow-1"), ("m-2", "mlflow-2")]


def test_training_logged_model_failure_only_warns(fake, client, tmp_path, monkeypatch, caplog):
    mlf = FakeMlflow(fail_create=True)
    with caplog.at_level(logging.WARNING):
        _log_model(monkeypatch, fake, client, tmp_path, mlf)                # does not raise
    assert mlf.logged_models == [] and mlf.metrics == []
    assert "kaos.logged_model_id" not in mlf.run_tags["mlflow-1"]
    assert "Failed to log model exp-best" in caplog.text


def test_training_without_an_mlflow_run_logs_no_model(fake, client, tmp_path, monkeypatch):
    mlf = FakeMlflow()
    _install(monkeypatch, mlf)
    mod = import_in("model_training", "app.services.model_training")
    svc = _training_service(mod, tmp_path, client, monkeypatch)
    svc._log_model("", "c0ffee", _params(), EPOCH_METRICS, 50)
    assert mlf.logged_models == [] and mlf.run_tags == {"mlflow-1": {}}


# ---------------------------------------------------------------- registration -> lakeFS tag

LINKED = {"kaos.outputs_repo": "outputs-yolo", "kaos.outputs_path": "checkpoints/exp/run-1/",
          "kaos.workflow": "wf-abc12"}


def _register(monkeypatch, mlf, name="yolo"):
    _install(monkeypatch, mlf)
    svc_mod = import_in("model_registration", "app.services.model_registration")
    reg = import_in("model_registration", "app.models.registration")
    params = reg.RegistrationParams(mlflow_run_id="mlflow-1", registered_model_name=name,
                                    best_checkpoint_path="s3://outputs-yolo/main/checkpoints/exp/run-1/best.pt")
    return svc_mod, svc_mod.ModelRegistrationService("http://mlflow", max_retries=1).run(params)


def test_lakefs_tag_id_slug_rules(monkeypatch):
    _install(monkeypatch, FakeMlflow())
    mod = import_in("model_registration", "app.services.model_registration")
    assert mod.lakefs_tag_id("yolo", 3) == "model-yolo-v3"
    assert mod.lakefs_tag_id("object-pose_yolo", 1) == "model-object-pose_yolo-v1"
    assert mod.lakefs_tag_id("team/My Model.v2", 12) == "model-team-My-Model-v2-v12"


def test_registration_tags_the_training_commit_and_records_it_on_the_version(fake, connected, monkeypatch):
    commit = _merged_commit(fake)
    mlf = FakeMlflow({**LINKED, "kaos.outputs_commit": commit})
    _, result = _register(monkeypatch, mlf)
    assert (result.best_version, result.last_version) == (1, 2)
    assert fake.repos["outputs-yolo"]["tags"] == {"model-yolo-v1": commit}, "one tag, for best only"
    best = mlf.version_tags[("yolo", 1)]
    assert {k: v for k, v in best.items() if k.startswith("kaos.")} == {
        "kaos.outputs_repo": "outputs-yolo", "kaos.outputs_commit": commit,
        "kaos.lakefs_tag": "model-yolo-v1", "kaos.workflow": "wf-abc12"}
    assert best["checkpoint_type"] == "best" and best["training_run_id"] == "mlflow-1"
    assert not any(k.startswith("kaos.") for k in mlf.version_tags[("yolo", 2)]), "last.pt: no link"


def test_registration_registers_best_from_the_logged_model_training_made(
        fake, connected, client, tmp_path, monkeypatch):
    commit = _merged_commit(fake)
    mlf = FakeMlflow({**LINKED, "kaos.outputs_commit": commit})
    _log_model(monkeypatch, fake, client, tmp_path, mlf)
    _, result = _register(monkeypatch, mlf)
    assert mlf.versions == [
        ("yolo", "models:/m-1"),
        ("yolo", "s3://outputs-yolo/main/checkpoints/exp/run-1/last.pt")], "last.pt as before"
    assert fake.repos["outputs-yolo"]["tags"] == {"model-yolo-v1": commit}
    best = mlf.version_tags[("yolo", 1)]
    assert best["kaos.lakefs_tag"] == "model-yolo-v1" and best["kaos.outputs_commit"] == commit
    assert best["checkpoint_type"] == "best" and best["training_run_id"] == "mlflow-1"


def test_registration_rerun_with_the_same_tag_and_commit_succeeds(fake, connected, monkeypatch):
    commit = _merged_commit(fake)
    fake.repos["outputs-yolo"]["tags"]["model-yolo-v1"] = commit
    mlf = FakeMlflow({**LINKED, "kaos.outputs_commit": commit})
    _register(monkeypatch, mlf)
    assert fake.repos["outputs-yolo"]["tags"] == {"model-yolo-v1": commit}
    assert mlf.version_tags[("yolo", 1)]["kaos.lakefs_tag"] == "model-yolo-v1"


def test_registration_fails_when_the_tag_points_at_another_commit(fake, connected, monkeypatch):
    other = fake.head("outputs-yolo")
    commit = _merged_commit(fake)
    fake.repos["outputs-yolo"]["tags"]["model-yolo-v1"] = other
    mlf = FakeMlflow({**LINKED, "kaos.outputs_commit": commit})
    with pytest.raises(Exception, match="model-yolo-v1.*already points at"):
        _register(monkeypatch, mlf)
    assert fake.repos["outputs-yolo"]["tags"] == {"model-yolo-v1": other}, "never moved"
    assert "kaos.lakefs_tag" not in mlf.version_tags.get(("yolo", 1), {})


def test_registration_of_an_unlinked_run_skips_the_tag_and_says_why(fake, connected, monkeypatch, caplog):
    mlf = FakeMlflow({"kubecore.run_id": "old-run"})       # a run from before kaos.outputs_commit existed
    with caplog.at_level(logging.WARNING):
        _, result = _register(monkeypatch, mlf)
    assert result.best_version == 1 and fake.repos["outputs-yolo"]["tags"] == {}
    assert mlf.versions[0] == ("yolo", "s3://outputs-yolo/main/checkpoints/exp/run-1/best.pt"), \
        "no logged model: registered from lakeFS as before"
    assert not any(k.startswith("kaos.") for k in mlf.version_tags[("yolo", 1)])
    assert "no kaos.outputs_commit" in caplog.text
    assert "no kaos.logged_model_id" in caplog.text

