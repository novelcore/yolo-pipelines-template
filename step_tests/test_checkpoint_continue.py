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
    assert 'step("model-training", gpu=True, needs=[load], hpc_time_limit="12h", resumable=True,' in src
