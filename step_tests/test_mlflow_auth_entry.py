"""model-training: a platform-supplied bearer token enables MLflow without the machine key."""
from conftest import import_in


def test_platform_token_enables_mlflow_without_machine_key(monkeypatch):
    entry = import_in("model_training", "app.entry")
    monkeypatch.delenv("ZITADEL_MACHINE_KEY_FILE", raising=False)
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "jwt")
    monkeypatch.setenv("MLFLOW_TRACKING_AUTH", "zitadel")
    assert entry._setup_mlflow_auth() is True
    assert "MLFLOW_TRACKING_AUTH" not in entry.os.environ
