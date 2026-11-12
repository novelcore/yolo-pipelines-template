"""qat-finetune resumes through kaos-lib's generic checkpoint folder (PRD-HPC-1461 D-76).

A tiny conv model stands in for YOLO: it is exported, prepared for QAT with torchao's PT2E flow (the same
``prepare_qat_pt2e`` the step uses, with torchao's x86 quantizer instead of litert_torch's, which only
changes which fake-quantize nodes are inserted), trained a few batches, saved and restored into a freshly
prepared module. The checkpoint folder is a temp dir named by KAOS_CHECKPOINT_DIR.
"""
import io
import json
import sys
import types

import pytest
from conftest import import_in

torch = pytest.importorskip("torch", reason="the QAT resume tests need torch (CPU is enough)")
pytest.importorskip("torchao.quantization.pt2e.quantize_pt2e", reason="the QAT resume tests need torchao")
from kaos_lib import checkpoint  # noqa: E402
from torchao.quantization.pt2e import allow_exported_model_train_eval  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import prepare_qat_pt2e  # noqa: E402
from torchao.quantization.pt2e.quantizer.x86_inductor_quantizer import (  # noqa: E402
    X86InductorQuantizer, get_default_x86_inductor_quantization_config)

SHAPE = (1, 3, 8, 8)


class Tiny(torch.nn.Module):
    """conv-bn-relu-conv; ``model[-1]`` mimics the Detect head the step flips to training mode."""

    def __init__(self):
        super().__init__()
        self.model = torch.nn.ModuleList([torch.nn.Conv2d(3, 4, 3), torch.nn.Conv2d(4, 2, 1)])
        self.bn = torch.nn.BatchNorm2d(4)

    def forward(self, x):
        return self.model[1](torch.relu(self.bn(self.model[0](x))))


def prepare(seed=0):
    torch.manual_seed(seed)
    module = Tiny().eval()
    exported = torch.export.export(module, (torch.zeros(*SHAPE),), strict=False).module()
    quantizer = X86InductorQuantizer().set_global(get_default_x86_inductor_quantization_config(is_qat=True))
    prepared = prepare_qat_pt2e(exported, quantizer)
    allow_exported_model_train_eval(prepared)
    prepared.train()
    for p in prepared.parameters():
        p.requires_grad_(True)
    return module, prepared


def train_some(prepared, optimizer, batches=4):
    for i in range(batches):
        torch.manual_seed(100 + i)
        loss = prepared(torch.randn(*SHAPE)).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@pytest.fixture
def folder(tmp_path, monkeypatch):
    path = tmp_path / "ckpt"
    monkeypatch.setenv("KAOS_CHECKPOINT_DIR", str(path))
    monkeypatch.delenv("MLFLOW_RUN_ID", raising=False)
    monkeypatch.setitem(checkpoint._state, "requested", False)
    return path


@pytest.fixture
def resume():
    return import_in("qat_finetune", "app.services.qat_resume")


def assert_same_tensors(a, b):
    assert list(a) == list(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_should_round_trip_prepared_state_dict_including_fake_quant_buffers_into_fresh_module(folder, resume):
    _, prepared = prepare(seed=0)
    optimizer = torch.optim.Adam(prepared.parameters(), lr=1e-2)
    train_some(prepared, optimizer)
    quant_buffers = [k for k in prepared.state_dict() if "scale" in k or "zero_point" in k or "min_val" in k]
    assert quant_buffers, "the prepared module must carry observer / fake-quantize buffers"

    resume.save_epoch(prepared, optimizer, 2)

    _, fresh = prepare(seed=1)                       # different init: nothing matches before restore
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-2)
    state = resume.load_state("cpu")
    assert resume.restore(fresh, fresh_opt, state) == 2

    assert_same_tensors(prepared.state_dict(), fresh.state_dict())
    probe = torch.randn(*SHAPE)
    prepared.eval(), fresh.eval()
    assert torch.equal(prepared(probe), fresh(probe))
    assert fresh_opt.state_dict()["param_groups"][0]["lr"] == 1e-2
    assert len(fresh_opt.state_dict()["state"]) == len(optimizer.state_dict()["state"]) > 0


def test_should_continue_training_identically_after_restore(folder, resume):
    _, prepared = prepare()
    optimizer = torch.optim.Adam(prepared.parameters(), lr=1e-2)
    train_some(prepared, optimizer, 2)
    resume.save_epoch(prepared, optimizer, 1)
    train_some(prepared, optimizer, 2)               # the uninterrupted run

    _, fresh = prepare(seed=3)
    fresh_opt = torch.optim.Adam(fresh.parameters(), lr=1e-2)
    resume.restore(fresh, fresh_opt, resume.load_state("cpu"))
    train_some(fresh, fresh_opt, 2)                  # the resumed run (same batches)

    assert_same_tensors(prepared.state_dict(), fresh.state_dict())


def test_should_report_epoch_progress_and_write_one_atomic_file(folder, resume):
    _, prepared = prepare()
    optimizer = torch.optim.Adam(prepared.parameters())
    resume.save_epoch(prepared, optimizer, 3)

    assert (folder / "qat.bin").is_file()
    assert not [p for p in folder.iterdir() if p.name.startswith(".tmp-")]
    assert json.loads((folder / "kaos-progress.json").read_text())["values"] == [3]


def test_should_start_from_epoch_zero_when_nothing_is_saved(folder, resume):
    assert resume.load_state("cpu") is None


@pytest.mark.parametrize("content", [b"not a torch file", b""])
def test_should_ignore_an_unreadable_checkpoint(folder, resume, content, capsys):
    folder.mkdir(parents=True)
    (folder / "qat.bin").write_bytes(content)
    assert resume.load_state("cpu") is None
    assert "starting from epoch 0" in capsys.readouterr().out


def test_should_reject_a_checkpoint_of_another_graph(folder, resume):
    _, prepared = prepare()
    optimizer = torch.optim.Adam(prepared.parameters())
    resume.save_epoch(prepared, optimizer, 1)
    other = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError):
        resume.restore(other, torch.optim.Adam(other.parameters()), resume.load_state("cpu"))


def test_should_load_a_tensor_only_file_never_arbitrary_pickles(folder, resume, capsys):
    import os
    buffer = io.BytesIO()
    torch.save({"format": 1, "epoch": 1, "model": {}, "optimizer": os.getcwd}, buffer)   # a function: not allowed
    checkpoint.save(buffer.getvalue(), "qat")
    assert resume.load_state("cpu") is None


def test_should_log_into_the_first_jobs_mlflow_run_when_continued(folder, resume):
    assert resume.reuse_saved_run() is None
    resume.remember_run("run-abc")
    assert json.loads((folder / "kaos-run.json").read_text()) == {"mlflow_run_id": "run-abc"}   # kaos-lib hook's file
    assert resume.reuse_saved_run() == "run-abc"
    import os
    assert os.environ["MLFLOW_RUN_ID"] == "run-abc"


def test_should_keep_a_run_id_the_platform_already_set(folder, resume, monkeypatch):
    resume.remember_run("run-abc")
    monkeypatch.setenv("MLFLOW_RUN_ID", "platform-run")
    assert resume.reuse_saved_run() == "platform-run"


def test_should_ignore_stop_until_an_epoch_is_saved(folder, resume, monkeypatch):
    monkeypatch.setitem(checkpoint._state, "requested", True)
    assert not resume.stop_wanted(0)
    assert resume.stop_wanted(1)


# ---------------------------------------------------------------- the real loop, with stubbed MLflow / YOLO imports

@pytest.fixture
def service(folder, monkeypatch):
    """QATService with MLflow and ultralytics stubbed (only imported, never used by _finetune)."""
    stub = types.ModuleType("mlflow")
    stub.log_metric = lambda *a, **k: None
    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = object
    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = object
    for name, mod in (("mlflow", stub), ("mlflow.tracking", tracking), ("ultralytics", ultra)):
        monkeypatch.setitem(sys.modules, name, mod)
    mod = import_in("qat_finetune", "app.services.qat_service")
    return mod.QATService()


def fixed_loader():
    gen = torch.Generator().manual_seed(7)
    return [torch.randn(*SHAPE, generator=gen) for _ in range(3)]


def run_finetune(service, params, prepared, teacher, monkeypatch):
    loader = fixed_loader()   # the real loader is deterministic too: fixed sample, no shuffle
    monkeypatch.setattr(service, "_build_calibration_loader", lambda *a, **k: loader)
    service._finetune(prepared, teacher, params, "cpu")


def params(epochs):
    return types.SimpleNamespace(qat_epochs=epochs, qat_lr=1e-2, dataset_dir="", image_size=8,
                                 calibration_frames=3, calibration_seed=0)


def test_should_resume_the_loop_from_the_saved_epoch_and_match_an_uninterrupted_run(service, folder, monkeypatch):
    # Uninterrupted: 3 epochs, no checkpoint reuse.
    teacher, reference = prepare()
    run_finetune(service, params(3), reference, teacher, monkeypatch)
    ref_state = {k: v.clone() for k, v in reference.state_dict().items()}
    for f in folder.iterdir():
        f.unlink()

    # Interrupted after 1 epoch: the stop arrives during epoch 2, which is dropped; the step exits with 85.
    teacher, first = prepare()
    seen = {"batches": 0}
    real = checkpoint.stop_requested

    def stop_in_epoch_two():
        seen["batches"] += 1
        return seen["batches"] > 4          # epoch 1 = 3 batches, then stop on the 2nd batch of epoch 2
    monkeypatch.setattr(checkpoint, "stop_requested", stop_in_epoch_two)
    with pytest.raises(SystemExit) as exit_info:
        run_finetune(service, params(3), first, teacher, monkeypatch)
    assert exit_info.value.code == checkpoint.CONTINUE_EXIT_CODE
    monkeypatch.setattr(checkpoint, "stop_requested", real)
    assert json.loads((folder / "kaos-progress.json").read_text())["values"] == [1]

    # The repeated step: a fresh prepared module continues with epochs 2 and 3 only.
    teacher, second = prepare(seed=9)
    run_finetune(service, params(3), second, teacher, monkeypatch)
    assert json.loads((folder / "kaos-progress.json").read_text())["values"] == [3]
    assert_same_tensors(ref_state, second.state_dict())


def test_should_skip_training_when_all_epochs_are_already_saved(service, folder, monkeypatch, resume):
    teacher, prepared = prepare()
    optimizer = torch.optim.Adam(prepared.parameters(), lr=1e-2)
    train_some(prepared, optimizer)
    resume.save_epoch(prepared, optimizer, 2)
    saved = {k: v.clone() for k, v in prepared.state_dict().items()}

    teacher, fresh = prepare(seed=4)
    run_finetune(service, params(2), fresh, teacher, monkeypatch)
    assert_same_tensors(saved, fresh.state_dict())
