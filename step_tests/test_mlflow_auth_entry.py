"""MLflow auth and the data lock come from kaos-lib[mlflow] (novelcore/kaos#1147 F-05, F-18).

No step carries its own copy of the `zitadel` auth plugin any more: kaos-lib registers the
request-auth provider and every step calls ``kaos_lib.mlflow.mlflow()``. Training and
registration record the dataset commit they used in the MLflow run (``kaos_lib.lock``).
"""
import json
import types
from pathlib import Path

import pytest
import requests
from conftest import import_in

ROOT = Path(__file__).resolve().parents[1]
MLFLOW_STEPS = ("model_training", "model_registration", "model_quantization", "qat_finetune")


@pytest.fixture(autouse=True)
def mlflow_env(monkeypatch):
    """No MLflow env from the caller; kaos-lib's token cache and data lock start empty."""
    from kaos_lib import lock
    from kaos_lib import mlflow as kmlflow
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_TRACKING_TOKEN", "MLFLOW_TRACKING_AUTH", "MLFLOW_RUN_ID",
                "ZITADEL_MACHINE_KEY_FILE", "ZITADEL_DOMAIN", "ZITADEL_MLFLOW_PROJECT_ID", "KAOS_LIB_LOCK"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KAOS_ML_HOME", "/nonexistent/kaos-login")
    monkeypatch.setenv("KUBECORE_ML_HOME", "/nonexistent/kaos-login")   # no laptop session either
    monkeypatch.setattr(kmlflow, "_source", kmlflow.TokenSource())
    lock.reset()
    yield
    lock.reset()


class FakeMlflowClient:
    """The calls kaos_lib.lock makes: set_tag, log_text, list_artifacts, download_artifacts."""

    def __init__(self, fail=None):
        self.tags, self.artifacts, self.fail = {}, {}, fail

    def set_tag(self, run_id, key, value):
        if self.fail:
            raise self.fail
        self.tags.setdefault(run_id, {})[key] = value

    def log_text(self, run_id, text, artifact_file):
        self.artifacts.setdefault(run_id, {})[artifact_file] = text

    def list_artifacts(self, run_id, path=None):
        if self.fail:
            raise self.fail
        return [types.SimpleNamespace(path=p) for p in self.artifacts.get(run_id, {})]

    def download_artifacts(self, run_id, path, dst_path=None):
        target = Path(dst_path) / path
        target.write_text(self.artifacts[run_id][path])
        return str(target)


@pytest.fixture
def mlflow_client(monkeypatch):
    from kaos_lib import lock
    client = FakeMlflowClient()
    monkeypatch.setattr(lock, "_client", lambda: client)
    return client


# ------------------------------------------------------------------ no copied plugin
@pytest.mark.parametrize("step", MLFLOW_STEPS)
def test_no_step_carries_its_own_auth_plugin(step):
    step_dir = ROOT / "steps" / step
    assert not (step_dir / "app" / "mlflow_zitadel_auth.py").exists()
    pyproject = (step_dir / "pyproject.toml").read_text()
    assert "mlflow.request_auth_provider" not in pyproject and "authlib" not in pyproject
    assert 'name = "authlib"' not in (step_dir / "poetry.lock").read_text()
    assert "mlflow_zitadel_auth" not in (step_dir / "app" / "entry.py").read_text()


def test_kaos_lib_registers_the_zitadel_provider():
    from importlib.metadata import entry_points
    found = {ep.name: ep.value for ep in entry_points(group="mlflow.request_auth_provider")}
    assert found.get("zitadel") == "kaos_lib.mlflow:ZitadelRequestAuthProvider"


# ------------------------------------------------------------------ _setup_mlflow_auth
@pytest.mark.parametrize("step", MLFLOW_STEPS)
def test_platform_token_enables_mlflow_without_machine_key(step, monkeypatch):
    """MeluXina: the platform hands the job MLFLOW_TRACKING_TOKEN and the public URI."""
    entry = import_in(step, "app.entry")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow-b.example")
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "jwt")
    assert entry._setup_mlflow_auth() is True
    assert entry.os.environ["MLFLOW_TRACKING_AUTH"] == "zitadel"   # the kaos-lib provider sends the bearer
    from kaos_lib.mlflow import ZitadelRequestAuthProvider
    req = ZitadelRequestAuthProvider().get_auth()(requests.Request("GET", "https://mlflow-b.example").prepare())
    assert req.headers["Authorization"] == "Bearer jwt"


@pytest.mark.parametrize("step", MLFLOW_STEPS)
def test_no_credential_disables_mlflow_without_failing(step, monkeypatch, capsys):
    entry = import_in(step, "app.entry")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.ml-b.svc:5000")
    assert entry._setup_mlflow_auth() is False
    assert "no MLflow credential" in capsys.readouterr().out


def test_machine_key_without_project_id_disables_mlflow(monkeypatch, tmp_path, capsys):
    """Project MLflow only accepts tokens audienced to the org project: no project id, no token."""
    entry = import_in("model_training", "app.entry")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.ml-b.svc:5000")
    monkeypatch.setenv("ZITADEL_MACHINE_KEY_FILE", str(tmp_path / "key.json"))
    monkeypatch.setenv("ZITADEL_DOMAIN", "zitadel.test")
    assert entry._setup_mlflow_auth() is False
    assert "ZITADEL_MLFLOW_PROJECT_ID not set" in capsys.readouterr().out


# ------------------------------------------------------------------ data lock
def test_training_writes_the_data_lock(mlflow_client):
    entry = import_in("model_training", "app.entry")
    assert entry._write_data_lock("mlflow-1", "dataset-cats@v1", "c0001abcd") is True
    assert mlflow_client.tags["mlflow-1"] == {"kaos_lib.data.dataset-cats.commit": "c0001abcd"}
    assert json.loads(mlflow_client.artifacts["mlflow-1"]["data.lock.json"]) == {
        "version": 1, "reads": [{"repo": "dataset-cats", "owner": None, "ref": "v1", "commit": "c0001abcd"}]}


def test_training_records_a_shared_dataset_with_its_owner(mlflow_client):
    entry = import_in("model_training", "app.entry")
    entry._write_data_lock("mlflow-1", "team-a/dataset-cats@v3", "c0003")
    assert mlflow_client.tags["mlflow-1"]["kaos_lib.data.dataset-cats.owner"] == "team-a"


def test_registration_adds_to_the_lock_training_wrote(mlflow_client):
    import_in("model_training", "app.entry")._write_data_lock("mlflow-1", "dataset-cats@v1", "c0001")
    from kaos_lib import lock
    lock.reset()                                             # registration runs in its own pod
    entry = import_in("model_registration", "app.entry")
    assert entry._write_data_lock("mlflow-1", "dataset-raw@main", "c0009") is True
    reads = json.loads(mlflow_client.artifacts["mlflow-1"]["data.lock.json"])["reads"]
    assert [(r["repo"], r["commit"]) for r in reads] == [("dataset-cats", "c0001"), ("dataset-raw", "c0009")]


@pytest.mark.parametrize("step", ["model_training", "model_registration"])
def test_data_lock_is_skipped_without_a_pinned_commit(step, mlflow_client):
    entry = import_in(step, "app.entry")
    assert entry._write_data_lock("mlflow-1", "dataset-cats@v1", "") is False
    assert mlflow_client.tags == {}


@pytest.mark.parametrize("step", ["model_training", "model_registration"])
def test_data_lock_failure_is_not_fatal(step, monkeypatch, capsys):
    from kaos_lib import lock
    monkeypatch.setattr(lock, "_client", lambda: FakeMlflowClient(fail=ConnectionError("mlflow down")))
    entry = import_in(step, "app.entry")
    assert entry._write_data_lock("mlflow-1", "dataset-cats@v1", "c0001") is False
    assert "data lock not written" in capsys.readouterr().out
