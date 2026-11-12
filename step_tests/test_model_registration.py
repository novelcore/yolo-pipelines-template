"""model-registration: a trained model is linked both ways between lakeFS (`outputs-<app>`) and MLflow (kaos#1283).

It reads where model-training saved its results from the step's upstream result (the repository and the commit),
shows best.pt in the experiment's Models tab as a logged model, registers best.pt (and last.pt) from it, tags the
commit in lakeFS (`model-<slug(name)>-v<N>`) and records the tag on the model version. MLflow is a recording fake.
"""
import importlib
import types
from importlib.metadata import version

import pytest
from conftest import composed_settings, step_app
from kaos_lib import testing
from kaos_lib.errors import NotFoundError
from kaos_lib.lakefs import LakeFS

COMMIT, QUANT_COMMIT = "c" * 40, "d" * 40


class FakeMlflow:
    """The slice of mlflow the step calls: get_run, logged models, register_model, tags, metrics, aliases."""

    EXPERIMENT_ID = "7"

    def __init__(self, fail_create=False):
        self.run_tags, self.version_tags, self.registered, self.aliases = {}, {}, [], {}
        self.logged_models, self.metrics, self.fail_create = {}, [], fail_create

    # mlflow module level
    def create_external_model(self, name, source_run_id, experiment_id, params, tags):
        if self.fail_create:
            raise RuntimeError("mlflow 3 only")
        self.logged_models[f"m-{len(self.logged_models) + 1}"] = {"name": name, "run": source_run_id, "params": params,
                                                                   "tags": dict(tags)}
        return types.SimpleNamespace(model_id=f"m-{len(self.logged_models)}")

    def register_model(self, model_uri, name):
        self.registered.append((name, model_uri))
        return types.SimpleNamespace(version=str(len(self.registered)))

    # MlflowClient
    def get_run(self, run_id):
        return types.SimpleNamespace(info=types.SimpleNamespace(experiment_id=self.EXPERIMENT_ID),
                                     data=types.SimpleNamespace(tags=self.run_tags))

    def search_logged_models(self, experiment_ids, filter_string, max_results):
        return [types.SimpleNamespace(model_id=i) for i, m in self.logged_models.items() if m["name"] in filter_string]

    def set_logged_model_tags(self, model_id, tags):
        self.logged_models[model_id]["tags"].update(tags)

    def log_metric(self, run_id, key, value, step, model_id):
        self.metrics.append((key, value, step, model_id))

    def set_tag(self, run_id, key, value):
        self.run_tags[key] = value

    def set_model_version_tag(self, name, version, key, value):
        self.version_tags.setdefault((name, int(version)), {})[key] = value

    def set_registered_model_alias(self, name, alias, version):
        self.aliases[(name, alias)] = int(version)


class FakeLake:
    def __init__(self):
        self.tags = {}

    def tag_commit(self, repo, tag):
        if (repo, tag) not in self.tags:
            raise NotFoundError(tag)
        return self.tags[(repo, tag)]

    def create_tag(self, repo, tag, ref):
        self.tags[(repo, tag)] = ref


@pytest.fixture
def modules():
    """The step's entry and its registration service, imported together (they share one `app`)."""
    with step_app("model_registration"):
        yield importlib.import_module("app.entry"), importlib.import_module("app.services.model_registration")


@pytest.fixture
def mlf(modules, monkeypatch):
    import mlflow
    fake = FakeMlflow()
    monkeypatch.setattr(mlflow, "create_external_model", fake.create_external_model, raising=False)
    monkeypatch.setattr(mlflow, "register_model", fake.register_model, raising=False)
    monkeypatch.setattr(modules[1], "MlflowClient", lambda: fake)
    return fake


@pytest.fixture
def lake(monkeypatch):
    fake = FakeLake()
    monkeypatch.setattr(LakeFS, "connect", classmethod(lambda cls, *a, **k: fake))
    return fake


@pytest.fixture
def entry(modules, monkeypatch):
    monkeypatch.setattr(modules[0].tracking, "mlflow", lambda *a, **k: None)  # signing in is kaos-lib's
    return modules[0]


TRAIN = {"best_checkpoint": "best.pt", "mlflow_run_id": "mlflow-1", "model_variant": "yolov8n-pose.pt",
         "epochs_completed": 3, "image_size": 640, "final_map50": 0.5, "final_map50_95": 0.3,
         "val_metrics": {"val/mAP50": 0.5, "val/precision": 0.9}, "exported_models": {}}
RAW = {"repo": "dataset-cats", "ref": "v2", "commit": "a" * 40}   # what prepare-data read
RUN = "test-run"                                                  # the run id kaos_lib.testing hands the step


def run(entry, tmp_path, train=None, quant=None, settings=None, name="model-registration"):
    upstream = {"prepare-data": {}, "model-training": {**TRAIN, **(train or {})},
                "model-quantization": quant["values"] if quant else None}
    writes = {"prepare-data": {"prepared": {"type": "dataset", "commit": "b" * 40}},
              "model-training": {"checkpoints": {"type": "model", "commit": COMMIT}}}
    if quant:
        writes["model-quantization"] = {"quantization": {"type": "model", "commit": quant["commit"]}}
    with testing.running(tmp_path, kind="publish", name=name, app="yolo", experiment="exp",
                         params=settings or composed_settings(), upstream=upstream, upstream_writes=writes,
                         upstream_reads={"prepare-data": {"dataset": RAW}}) as ran:
        entry.main()
    return ran.result


def test_registers_best_from_the_logged_model_and_last_from_its_pinned_path(entry, mlf, lake, tmp_path):
    out = run(entry, tmp_path)
    assert [(n, u) for n, u in mlf.registered] == [
        ("object-pose-yolo", "models:/m-1"),
        ("object-pose-yolo", f"s3://outputs-yolo/{COMMIT}/checkpoints/exp/{RUN}/last.pt")]
    assert out["values"]["best_version"] == 1 and out["values"]["last_version"] == 2


def test_the_logged_model_names_where_the_results_are_and_carries_the_val_metrics(entry, mlf, lake, tmp_path):
    run(entry, tmp_path)
    model = mlf.logged_models["m-1"]
    assert model["name"] == "exp-best" and model["run"] == "mlflow-1"
    assert model["tags"] == {"kaos.outputs_repo": "outputs-yolo", "kaos.outputs_commit": COMMIT,
                             "kaos.outputs_path": f"checkpoints/exp/{RUN}/", "kaos.workflow": RUN,
                             "kaos.dataset_ref": "dataset-cats@v2"}
    assert model["params"] == {"epochs": "3", "imgsz": "640", "model_variant": "yolov8n-pose.pt"}
    assert sorted((k, s, m) for k, _, s, m in mlf.metrics) == [("val/mAP50", 3, "m-1"), ("val/precision", 3, "m-1")]
    assert mlf.run_tags["kaos.logged_model_id"] == "m-1" and mlf.run_tags["kaos.outputs_commit"] == COMMIT


def test_a_run_that_has_its_logged_model_already_reuses_it(entry, mlf, lake, tmp_path):
    run(entry, tmp_path / "a")
    run(entry, tmp_path / "b")
    assert len(mlf.logged_models) == 1


def test_the_commit_is_tagged_in_lakefs_and_the_tag_is_recorded_on_the_version(entry, mlf, lake, tmp_path):
    run(entry, tmp_path)
    assert lake.tags == {("outputs-yolo", "model-object-pose-yolo-v1"): COMMIT}  # the best version
    assert mlf.version_tags[("object-pose-yolo", 1)]["kaos.lakefs_tag"] == "model-object-pose-yolo-v1"
    assert mlf.version_tags[("object-pose-yolo", 1)]["kaos.outputs_commit"] == COMMIT


def test_a_tag_that_points_elsewhere_is_never_moved(entry, mlf, lake, tmp_path):
    lake.tags[("outputs-yolo", "model-object-pose-yolo-v1")] = "e" * 40
    with pytest.raises(Exception, match="refusing to move it"):
        run(entry, tmp_path)


def test_the_versions_say_where_the_data_came_from_and_which_kaos_lib_trained_them(entry, mlf, lake, tmp_path):
    run(entry, tmp_path, settings=composed_settings(data={"sample_size": "50", "seed": 1}))
    tags = mlf.version_tags[("object-pose-yolo", 1)]
    assert tags["dataset_version"] == "dataset-cats@v2" and tags["dataset_commit"] == "a" * 40
    assert tags["prepared_dataset"] == f"dataset-yolo-prepared@{'b' * 40}" and tags["dataset_sample_size"] == "50"
    assert tags["training_run_id"] == "mlflow-1" and tags["best_mAP50"] == "0.5"
    assert tags["kaos_lib.version"] == version("kaos-lib")


def test_a_model_trained_to_last_pt_registers_last_only_once(entry, mlf, lake, tmp_path):
    run(entry, tmp_path, train={"best_checkpoint": "last.pt"})
    assert [u for _, u in mlf.registered] == ["models:/m-1"]


def test_exports_and_the_int8_model_are_recorded_with_the_commit_that_holds_them(entry, mlf, lake, tmp_path):
    quant = {"values": {"tflite": "model_int8.tflite", "mlflow_run_id": "mlflow-2"}, "commit": QUANT_COMMIT}
    run(entry, tmp_path, train={"exported_models": {"onnx_fp16": "best.onnx"}}, quant=quant)
    tags = mlf.version_tags[("object-pose-yolo", 1)]
    assert tags["export_onnx_fp16"] == f"s3://outputs-yolo/{COMMIT}/checkpoints/exp/{RUN}/best.onnx"
    assert tags["export_int8_tflite"] == f"s3://outputs-yolo/{QUANT_COMMIT}/quantization/exp/{RUN}/model_int8.tflite"


def test_the_alias_asked_for_is_given_to_the_best_version(entry, mlf, lake, tmp_path):
    settings = composed_settings(registration={"registered_model_name": "mine", "promote_to": "champion"})
    out = run(entry, tmp_path, settings=settings)
    assert mlf.aliases == {("mine", "champion"): 1} and out["values"]["promoted_to"] == "champion"


def test_an_mlflow_that_cannot_log_the_model_fails_the_step_and_says_so(entry, modules, mlf, lake, tmp_path, monkeypatch):
    mlf.fail_create = True
    monkeypatch.setattr(modules[1].ModelRegistrationService, "_RETRY_DELAYS", (0, 0, 0))
    with pytest.raises(Exception, match="MLflow call failed after 3 attempts: mlflow 3 only"):
        run(entry, tmp_path)
    assert mlf.registered == []
