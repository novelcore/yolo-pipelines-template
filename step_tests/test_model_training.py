"""model-training reads exactly the dataset version prepare-data wrote, and saves its weights as the model `checkpoints`."""
import types
from pathlib import Path

import pytest
from conftest import ROOT, composed_settings, import_in, make_dataset
from kaos_lib import testing


@pytest.fixture
def training():
    return import_in("model_training", "app.services.model_training")


@pytest.fixture
def entry():
    return import_in("model_training", "app.entry")


# ------------------------------------------------------------------------------------------- the step's inputs

def test_the_entry_hands_the_prepared_dataset_and_its_own_output_to_the_trainer(entry, tmp_path, monkeypatch):
    prepared = make_dataset(tmp_path / "prepared")
    seen = {}

    class Trainer:
        def __init__(self, level):
            seen["level"] = level

        def run(self, **kwargs):
            seen.update(kwargs)
            (Path(kwargs["checkpoints_dir"]) / "best.pt").write_bytes(b"w")
            return types.SimpleNamespace(model_dump=lambda: {"best_checkpoint": "best.pt", "mlflow_run_id": "mlflow-1"})

    monkeypatch.setattr(entry, "Manager", Trainer)
    monkeypatch.setattr(entry.tracking, "mlflow", lambda *a, **k: None)  # signing in is kaos-lib's
    with testing.running(tmp_path / "run", kind="train", name="model-training", experiment="object-pose-v1-yolov8n",
                         params=composed_settings(),
                         inputs={"prepared": prepared}, produces=[("model", "checkpoints")]) as run:
        entry.main()
    data = Path(seen["dataset_dir"])
    assert (data / "data.yaml").is_file(), "training finds data.yaml through step.input('prepared')"
    assert len(list((data / "images" / "train").iterdir())) == 6
    assert seen["experiment_name"] == "object-pose-v1-yolov8n" and seen["run_name"] == "test-run"
    assert seen["epochs"] == 100 and seen["batch_size"] == 16 and seen["level"] == "INFO"
    assert run.result["values"] == {"best_checkpoint": "best.pt", "mlflow_run_id": "mlflow-1"}
    assert (run.output("checkpoints") / "best.pt").is_file()


def test_the_entry_reads_the_dataset_through_the_step_api_only():
    text = (ROOT / "steps/model_training/app/entry.py").read_text()
    assert 'step.input("prepared")' in text and "datasets." not in text and "DatasetRef" not in text


# ------------------------------------------------------------------------------------------- the weights it hands on

def make_service(training):
    return training.TrainingService()


def params(tmp_path):
    out = tmp_path / "checkpoints"
    out.mkdir(exist_ok=True)
    return types.SimpleNamespace(checkpoints_dir=str(out))


def weights_dir(tmp_path, *names):
    folder = tmp_path / "run" / "weights"
    folder.mkdir(parents=True)
    for name in names:
        (folder / name).write_bytes(b"w")
    return tmp_path / "run"


def test_should_return_best_pt_and_save_both_when_both_exist(training, tmp_path):
    p = params(tmp_path)
    assert make_service(training)._collect_weights(p, weights_dir(tmp_path, "best.pt", "last.pt")) == "best.pt"
    assert sorted(f.name for f in (tmp_path / "checkpoints").iterdir()) == ["best.pt", "last.pt"]


def test_should_fall_back_to_last_pt_when_a_continued_run_has_no_best_pt(training, tmp_path):
    p = params(tmp_path)
    assert make_service(training)._collect_weights(p, weights_dir(tmp_path, "last.pt")) == "last.pt"
    assert [f.name for f in (tmp_path / "checkpoints").iterdir()] == ["last.pt"]


def test_should_raise_instead_of_returning_an_empty_name_when_no_weights_exist(training, tmp_path):
    p = params(tmp_path)
    with pytest.raises(training.TrainingError, match="no weights"):
        make_service(training)._collect_weights(p, weights_dir(tmp_path))
    assert list((tmp_path / "checkpoints").iterdir()) == []


def test_pretrained_weights_in_lakefs_are_downloaded_from_the_ref_they_name(training, tmp_path, monkeypatch):
    from kaos_lib.lakefs import LakeFS

    asked = []

    class Lake:
        def download_object(self, repo, ref, path, dest):
            asked.append((repo, ref, path))
            dest.write_bytes(b"w")

    monkeypatch.setattr(LakeFS, "connect", classmethod(lambda cls, *a, **k: Lake()))
    got = make_service(training)._fetch_weights("s3://outputs-yolo/abc123/checkpoints/exp/run/best.pt", tmp_path)
    assert asked == [("outputs-yolo", "abc123", "checkpoints/exp/run/best.pt")]
    assert got.name == "best.pt" and got.read_bytes() == b"w"


def test_a_missing_local_pretrained_weights_file_is_named(training, tmp_path):
    with pytest.raises(training.TrainingError, match="does not exist"):
        make_service(training)._fetch_weights(str(tmp_path / "gone.pt"), tmp_path)


def test_training_tags_its_mlflow_run_with_the_kaos_lib_version(training, monkeypatch):
    from importlib.metadata import version
    tags = {}

    class Client:
        def set_tag(self, run_id, key, value):
            tags[(run_id, key)] = value

    import mlflow.tracking
    monkeypatch.setattr(mlflow.tracking, "MlflowClient", Client)
    training.TrainingService._tag_run("mlflow-1")
    assert tags == {("mlflow-1", "kaos_lib.version"): version("kaos-lib")}
