"""On the stop signal before a Slurm time limit, the run writer saves Ultralytics' newest last.pt
under the name the resumed step looks for (kaos-lib autosave, PRD-HPC-1461 D-44)."""
import os
import types

import pytest

from conftest import step_app


def _service():
    with step_app("model_training"):
        from app.services.model_training import TrainingService
        return TrainingService.__new__(TrainingService)


def test_the_newest_last_pt_is_saved_on_the_stop_signal_and_found_on_resume(fake, tmp_path):
    from kaos_lib import outputs
    if not hasattr(outputs, "CHECKPOINT_EXIT_CODE"):
        pytest.skip("kaos-lib without the stop-signal save")
    svc = _service()
    weights = tmp_path / "runs" / "weights"
    weights.mkdir(parents=True)
    last = weights / "last.pt"
    last.write_bytes(b"epoch-3")
    with pytest.raises(SystemExit) as stopped:
        with outputs.run("yolo", "e", "r1", client=fake.client()) as w:
            svc._run_writer = w
            svc._make_autosave_callback()(types.SimpleNamespace(last=str(last)))
            last.write_bytes(b"epoch-9")                    # Ultralytics went on training
            os.kill(os.getpid(), outputs.STOP_SIGNAL)
    assert stopped.value.code == 85
    with outputs.run("yolo", "e", "r1", client=fake.client()) as w:
        assert w.latest("last.pt", dest=tmp_path / "resume").read_bytes() == b"epoch-9"


def test_an_older_kaos_lib_without_autosave_is_left_alone():
    svc = _service()
    svc._run_writer = types.SimpleNamespace()
    svc._make_autosave_callback()(types.SimpleNamespace(last="/x/last.pt"))   # no error
