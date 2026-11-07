"""qat-finetune: save(folder) -> {"epoch": n} / load(folder) restore the model and optimizer; anything
unusable or from a different run means a fresh start."""
import os

import pytest

from conftest import step_app

torch = pytest.importorskip("torch")


def _make(epochs=10, run_id="run-1"):
    with step_app("qat_finetune"):
        from app.services.qat_resume import QatResume
    model = torch.nn.Linear(3, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    r = QatResume(epochs, run_id)
    r.prepared, r.optimizer = model, opt
    return r, model, opt


def _train_a_bit(model, opt):
    model(torch.ones(1, 3)).sum().backward()
    opt.step()


def test_save_returns_progress_and_load_restores_model_and_optimizer(tmp_path):
    r, model, opt = _make()
    _train_a_bit(model, opt)
    r.epochs_done = 4
    assert r.save(tmp_path) == {"epoch": 4}
    fresh, model2, opt2 = _make()
    assert fresh.load(tmp_path) == 4
    assert torch.equal(model.weight, model2.weight)
    assert opt2.state_dict()["state"], "the optimizer state came back"


def test_a_checkpoint_of_another_epoch_budget_is_not_used(tmp_path):
    r, *_ = _make(epochs=10)
    r.epochs_done = 3
    r.save(tmp_path)
    other, model2, _ = _make(epochs=20)
    before = model2.weight.clone()
    assert other.load(tmp_path) == 0
    assert torch.equal(model2.weight, before)


def test_a_missing_or_corrupt_checkpoint_starts_fresh(tmp_path):
    r, *_ = _make()
    assert r.load(tmp_path) == 0
    (tmp_path / "state.json").write_text('{"epoch": 2, "qat_epochs": 10}')
    (tmp_path / "qat_state.pt").write_bytes(b"not a checkpoint")
    assert r.load(tmp_path) == 0


def test_the_continued_job_logs_to_the_same_mlflow_run(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    r, *_ = _make(run_id="run-xyz")
    r.epochs_done = 1
    r.save(tmp_path)
    with step_app("qat_finetune"):
        from app.services.qat_resume import adopt_mlflow_run
    adopt_mlflow_run(tmp_path)
    assert os.environ.pop("MLFLOW_RUN_ID") == "run-xyz"
