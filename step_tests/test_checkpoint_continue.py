"""model-training is resumable (pipeline.py): asked to stop before an HPC (Slurm) wall time, it saves Ultralytics'
newest last.pt with kaos_lib.checkpoint at the end of an epoch and ends with 85; the continued job resumes from it."""
import os
import types

import pytest

from conftest import step_app

checkpoint = pytest.importorskip("kaos_lib.checkpoint")


def _service():
    with step_app("model_training"):
        from app.services.model_training import TrainingService
        return TrainingService.__new__(TrainingService)


@pytest.fixture(autouse=True)
def clean(tmp_path, monkeypatch):
    checkpoint._state["requested"] = False
    monkeypatch.setenv(checkpoint.OUTPUT_DIR_ENV, str(tmp_path / "output"))
    monkeypatch.delenv(checkpoint.RESUME_PATH_ENV, raising=False)
    yield
    checkpoint._state["requested"] = False


def _last(tmp_path, data=b"epoch-9"):
    weights = tmp_path / "runs" / "weights"
    weights.mkdir(parents=True)
    (weights / "last.pt").write_bytes(data)
    return types.SimpleNamespace(last=str(weights / "last.pt"))


def test_without_a_stop_training_runs_on(tmp_path):
    _service()._make_stop_callback()(_last(tmp_path))
    assert not (tmp_path / "output").exists()


def test_asked_to_stop_it_saves_the_newest_last_pt_and_ends_with_85(tmp_path):
    trainer = _last(tmp_path)
    os.kill(os.getpid(), checkpoint.STOP_SIGNAL)
    with pytest.raises(SystemExit) as stopped:
        _service()._make_stop_callback()(trainer)
    assert stopped.value.code == 85
    assert (tmp_path / "output" / "checkpoint" / "last.pt").read_bytes() == b"epoch-9"


def test_asked_to_stop_before_the_first_last_pt_it_trains_on(tmp_path):
    os.kill(os.getpid(), checkpoint.STOP_SIGNAL)
    _service()._make_stop_callback()(types.SimpleNamespace(last=str(tmp_path / "missing.pt")))
    assert not (tmp_path / "output" / "checkpoint").exists()


def test_the_continued_job_resumes_from_the_saved_last_pt(tmp_path, monkeypatch):
    handed = tmp_path / "resume"
    handed.mkdir()
    (handed / "last.pt").write_bytes(b"epoch-9")
    monkeypatch.setenv(checkpoint.RESUME_PATH_ENV, str(handed))
    svc = _service()
    lakefs_io = type(svc)._open_run_writer.__globals__["lakefs_io"]     # the module the service itself uses
    monkeypatch.setattr(lakefs_io, "run_writer", lambda bucket, experiment: types.SimpleNamespace(
        latest=lambda *a, **k: pytest.fail("the run branch's older last.pt must not win")))
    svc._logger = types.SimpleNamespace(info=lambda *a: None)
    params = types.SimpleNamespace(checkpoint_bucket="outputs-yolo", experiment_name="e")
    assert svc._open_run_writer(params, tmp_path).read_bytes() == b"epoch-9"


def test_the_training_step_is_resumable():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[1] / "pipeline.py").read_text()
    assert 'step("model-training", gpu=True, needs=[load], hpc_time_limit="12h", resumable="30m",' in src
    assert 'disk="20Gi", resumable=True,' in src


# ---- save(folder) -> progress / load(folder): the shape the platform's checkpoint registration expects

def _trainer(tmp_path, epoch=4, epochs=100, stop=False, data=b"epoch-5"):
    weights = tmp_path / "runs" / "weights"
    weights.mkdir(parents=True, exist_ok=True)
    (weights / "last.pt").write_bytes(data)
    return types.SimpleNamespace(last=str(weights / "last.pt"), epoch=epoch, epochs=epochs, stop=stop)


def _resume():
    with step_app("model_training"):
        from app.services.resume import YoloResume
        return YoloResume()


def test_save_writes_last_pt_and_returns_the_epochs_done(tmp_path):
    r = _resume()
    r.trainer = _trainer(tmp_path, epoch=4)
    folder = tmp_path / "f"
    folder.mkdir()
    assert r.save(folder) == {"epoch": 5}
    assert (folder / "last.pt").read_bytes() == b"epoch-5"


def test_save_refuses_when_there_is_no_last_pt(tmp_path):
    r = _resume()
    r.trainer = types.SimpleNamespace(last=str(tmp_path / "none.pt"), epoch=0)
    with pytest.raises(RuntimeError):
        r.save(tmp_path)


def test_load_starts_fresh_when_the_folder_has_no_usable_last_pt(tmp_path):
    r = _resume()
    assert r.load(tmp_path) is None
    (tmp_path / "last.pt").write_bytes(b"")
    assert r.load(tmp_path) is None, "an empty file is never offered as a checkpoint"


def test_load_continues_the_first_jobs_mlflow_run(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    (tmp_path / "last.pt").write_bytes(b"x")
    (tmp_path / "state.json").write_text('{"epoch": 5, "mlflow_run_id": "abc123"}')
    assert _resume().load(tmp_path) == tmp_path / "last.pt"
    assert os.environ["MLFLOW_RUN_ID"] == "abc123"
    monkeypatch.delenv("MLFLOW_RUN_ID")


def test_a_checkpoint_without_state_json_still_loads_without_touching_mlflow(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    (tmp_path / "last.pt").write_bytes(b"x")
    assert _resume().load(tmp_path) is not None
    assert "MLFLOW_RUN_ID" not in os.environ


@pytest.mark.parametrize("trainer_kw", [dict(epoch=99, epochs=100), dict(epoch=3, epochs=100, stop=True)],
                         ids=["last_epoch", "early_stopping"])
def test_no_checkpoint_is_taken_when_training_ends_by_itself(tmp_path, trainer_kw):
    os.kill(os.getpid(), checkpoint.STOP_SIGNAL)
    _service()._make_stop_callback()(_trainer(tmp_path, **trainer_kw))      # returns: training finishes
    assert not (tmp_path / "output").exists()


def test_a_continuation_without_a_checkpoint_folder_starts_fresh(tmp_path):
    svc = _service()
    lakefs_io = type(svc)._open_run_writer.__globals__["lakefs_io"]
    svc._logger = types.SimpleNamespace(info=lambda *a: None)
    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(lakefs_io, "run_writer", lambda b, e: types.SimpleNamespace(latest=lambda *a, **k: None))
    try:
        params = types.SimpleNamespace(checkpoint_bucket="outputs-yolo", experiment_name="e")
        assert svc._open_run_writer(params, tmp_path) is None
    finally:
        mp.undo()
