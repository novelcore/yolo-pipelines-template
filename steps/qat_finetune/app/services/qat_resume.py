"""Resume support for the QAT fine-tune, on the platform's checkpoint folder (kaos-lib ``checkpoint``).

The fine-tune is a custom PyTorch loop, so kaos-lib's automatic Ultralytics hook does not apply. This module
is the whole of the step's resume code: it only chooses *what* to put in the folder (``checkpoint.dir()``);
keeping the folder safe and restoring it before the step starts again is the platform's job.

Two files are written, both atomically by ``checkpoint.save``:

* ``qat.bin``       the prepared module's state_dict (weights AND fake-quantize scales / observer buffers),
                    the optimizer state and the number of finished epochs, in one ``torch.save`` blob.
* ``kaos-run.json`` ``{"mlflow_run_id": ...}``, the same file name and shape kaos-lib's Ultralytics hook uses,
                    so the continued job logs into the first job's MLflow run.

Why one ``torch.save`` blob (bytes) rather than letting ``checkpoint.save`` pickle a dict of tensors:
tensors are moved to the CPU first, so a job that continues on a different kind of node (a GPU job continued on
a CPU node, another GPU index) can load it; ``torch.load(..., weights_only=True)`` reads only tensors and plain
containers, never arbitrary code; the state_dict keeps its ``_metadata`` (observers need it to load ``eps`` correctly); and weights, optimizer and epoch counter live in a single file, so they can
never disagree after a crash between two writes.
"""
from __future__ import annotations

import io
import os
from collections import OrderedDict
from typing import Any, Optional

import torch
from kaos_lib import checkpoint

STATE_NAME = "qat"            # -> qat.bin
RUN_NAME = "kaos-run"         # -> kaos-run.json (same file the kaos-lib Ultralytics hook writes)
FORMAT_VERSION = 1
_REQUIRED_KEYS = ("format", "epoch", "model", "optimizer")


def save_epoch(prepared: torch.nn.Module, optimizer: torch.optim.Optimizer, epochs_done: int) -> None:
    """Write the state after ``epochs_done`` finished epochs and report the epoch to the platform.

    Called at the end of every epoch (the safe point). The write is atomic, so a node lost mid-save leaves the
    previous epoch's file intact.
    """
    buffer = io.BytesIO()
    torch.save(
        {
            "format": FORMAT_VERSION,
            "epoch": int(epochs_done),
            "model": _to_cpu(prepared.state_dict()),
            "optimizer": _to_cpu(optimizer.state_dict()),
        },
        buffer,
    )
    checkpoint.save(buffer.getvalue(), STATE_NAME)
    checkpoint.progress({"epoch": int(epochs_done)})


def load_state(device: str) -> Optional[dict]:
    """The saved state mapped onto ``device``, or None on a first run or when the file is unusable.

    An unreadable or incomplete file is reported and ignored: the fine-tune then starts from epoch 0 rather
    than failing the step.
    """
    raw = checkpoint.load(STATE_NAME, default=None)
    if raw is None:
        return None
    try:
        state = torch.load(io.BytesIO(raw), map_location=device, weights_only=True)
        if not all(key in state for key in _REQUIRED_KEYS) or state["format"] != FORMAT_VERSION:
            raise ValueError("unexpected content")
        return state
    except Exception as exc:  # noqa: BLE001 - a bad checkpoint must not fail the step
        print(f"[qat-resume] ignoring the saved QAT state ({exc}); starting from epoch 0", flush=True)
        return None


def restore(prepared: torch.nn.Module, optimizer: torch.optim.Optimizer, state: dict) -> int:
    """Load ``state`` into a freshly prepared module and its optimizer; return the epoch to start from.

    ``strict=True``: a checkpoint from a different graph (other model or image size) raises instead of
    silently training half-restored weights.
    """
    prepared.load_state_dict(state["model"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    return int(state["epoch"])


def remember_run(mlflow_run_id: str) -> None:
    """Record the MLflow run this job logs into, so a continued job can log into the same one."""
    checkpoint.save({"mlflow_run_id": mlflow_run_id}, RUN_NAME)


def reuse_saved_run() -> Optional[str]:
    """Point MLflow at the first job's run, as kaos-lib's Ultralytics hook does (``MLFLOW_RUN_ID``).

    MLflow's ``start_run`` resumes the run named by ``MLFLOW_RUN_ID``. A value already set by the platform wins.
    Returns the run id in effect, or None on a first run.
    """
    saved = checkpoint.load(RUN_NAME, default=None)
    run_id = saved.get("mlflow_run_id") if isinstance(saved, dict) else None
    if run_id:
        os.environ.setdefault("MLFLOW_RUN_ID", run_id)
    return os.environ.get("MLFLOW_RUN_ID") or None


def stop_wanted(epochs_done: int) -> bool:
    """True when the platform asked the step to stop AND there is a saved epoch to continue from.

    With nothing saved yet a stop is ignored until the first epoch has been written, so a continuation
    always has something to continue from.
    """
    return checkpoint.stop_requested() and epochs_done > 0


def _to_cpu(obj: Any) -> Any:
    """Copy every tensor in a (nested) state dict to the CPU; leave other values as they are."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, OrderedDict):
        # A module state_dict carries ``_metadata`` (per-module format versions) as an attribute. Observers
        # read it on load: without it they take an old-format path that resets ``eps`` to the float32 epsilon.
        copy = OrderedDict((key, _to_cpu(value)) for key, value in obj.items())
        if hasattr(obj, "_metadata"):
            copy._metadata = obj._metadata
        return copy
    if isinstance(obj, dict):
        return {key: _to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(value) for value in obj)
    return obj
