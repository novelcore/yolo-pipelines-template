"""Save and load of a training run's state, for steps that continue after an HPC (Slurm) wall time.

``save(folder)`` and ``load(folder)`` are plain functions with the shape the platform's checkpoint
registration expects: ``save`` writes the state into an empty folder and returns the progress made
(``{"epoch": n}``); ``load`` reads it back. They know nothing about signals or exit codes.

Today the step calls ``kaos_lib.checkpoint.save(resume.save)`` at the end of an epoch once a stop was
requested. When kaos-lib offers ``kaos.checkpoint(save=..., load=...)``, replace that call in
``model_training._make_stop_callback`` by one registration of these two functions; nothing else changes.

Ultralytics' own ``last.pt`` holds the whole trainer state (weights, optimizer, EMA, epoch, the
epoch budget), so continuing is ``resume=True`` on that file and the total epochs stay the ones of the first job.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

_logger = logging.getLogger(__name__)

LAST_PT = "last.pt"
STATE_JSON = "state.json"
MLFLOW_RUN_ENV = "MLFLOW_RUN_ID"      # read by mlflow.start_run(): the continued job logs to the same run


class YoloResume:
    """Holds the running trainer so ``save`` can be a plain ``save(folder)`` function."""

    def __init__(self) -> None:
        self.trainer: Any = None

    def save(self, folder: Path) -> dict:
        """Write ``last.pt`` and the MLflow run id into ``folder``; return ``{"epoch": <epochs done>}``."""
        last = Path(getattr(self.trainer, "last", "") or "")
        if not last.is_file() or last.stat().st_size == 0:
            raise RuntimeError(f"no last.pt to save at {last}")
        shutil.copy2(last, folder / LAST_PT)
        epoch = int(getattr(self.trainer, "epoch", -1)) + 1      # Ultralytics' epoch is 0-based
        (folder / STATE_JSON).write_text(json.dumps({"epoch": epoch, "mlflow_run_id": _mlflow_run_id()}))
        return {"epoch": epoch}

    def load(self, folder: Path) -> Optional[Path]:
        """Return the ``last.pt`` to resume from, or None (start fresh) when the folder holds no usable one.

        Also points MLflow at the run the first job opened, so metrics continue in one run.
        """
        last = Path(folder) / LAST_PT
        if not last.is_file() or last.stat().st_size == 0:
            return None
        run_id = _read_run_id(Path(folder) / STATE_JSON)
        if run_id:
            os.environ[MLFLOW_RUN_ENV] = run_id
        return last


def training_is_ending(trainer: Any) -> bool:
    """True when the trainer stops by itself after this epoch (last epoch, early stopping, time limit).

    Saving then would hand a finished run to a continuation, and Ultralytics refuses to resume it.
    """
    return bool(getattr(trainer, "stop", False)) or int(getattr(trainer, "epoch", 0)) + 1 >= int(
        getattr(trainer, "epochs", 1 << 30))


def _mlflow_run_id() -> str:
    try:
        import mlflow  # noqa: PLC0415
        run = mlflow.active_run()
        return run.info.run_id if run else ""
    except Exception:  # noqa: BLE001 - MLflow is optional
        return ""


def _read_run_id(path: Path) -> str:
    try:
        return str(json.loads(path.read_text()).get("mlflow_run_id") or "")
    except (OSError, ValueError):
        return ""
