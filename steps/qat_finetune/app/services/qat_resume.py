"""Save and load of the QAT fine-tune state, for a step that continues after an HPC (Slurm) wall time.

``save(folder)`` writes the fake-quantized model, the optimizer and the number of finished epochs into an
empty folder and returns the progress (``{"epoch": n}``); ``load(folder)`` restores them and returns the
epoch to start from. They know nothing about signals or exit codes. Today the fine-tune loop calls
``kaos_lib.checkpoint.save(resume.save)`` at the end of an epoch once a stop was requested; when kaos-lib
offers ``kaos.checkpoint(save=..., load=...)`` it registers these two functions instead (see ``_finetune``).

A checkpoint is only used when it matches this run (same epoch budget); anything unreadable or different
means a fresh start, never a partly loaded model.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

STATE_FILE = "qat_state.pt"
META_FILE = "state.json"
MLFLOW_RUN_ENV = "MLFLOW_RUN_ID"      # read by mlflow.start_run(): the continued job logs to the same run


class QatResume:
    """Holds the live model, optimizer and epoch counter so ``save`` can be a plain ``save(folder)``."""

    def __init__(self, qat_epochs: int, mlflow_run_id: str = "") -> None:
        self.qat_epochs = qat_epochs
        self.mlflow_run_id = mlflow_run_id
        self.prepared: Any = None
        self.optimizer: Any = None
        self.epochs_done = 0

    def save(self, folder: Path) -> dict:
        """Write the state into ``folder``; return ``{"epoch": <epochs finished>}``."""
        import torch  # noqa: PLC0415
        torch.save({"prepared": self.prepared.state_dict(), "optimizer": self.optimizer.state_dict()},
                   Path(folder) / STATE_FILE)
        (Path(folder) / META_FILE).write_text(json.dumps(
            {"epoch": self.epochs_done, "qat_epochs": self.qat_epochs, "mlflow_run_id": self.mlflow_run_id}))
        return {"epoch": self.epochs_done}

    def load(self, folder: Path) -> int:
        """Restore model and optimizer from ``folder``; return the epoch to start from (0 = fresh start)."""
        try:
            meta = json.loads((Path(folder) / META_FILE).read_text())
            if int(meta["qat_epochs"]) != self.qat_epochs:
                _logger.warning("QAT checkpoint was made for %s epochs, this run has %s: starting fresh",
                                meta["qat_epochs"], self.qat_epochs)
                return 0
            import torch  # noqa: PLC0415
            state = torch.load(Path(folder) / STATE_FILE, map_location="cpu", weights_only=True)
            self.prepared.load_state_dict(state["prepared"])
            self.optimizer.load_state_dict(state["optimizer"])
        except Exception as exc:  # noqa: BLE001 - an unusable checkpoint must not stop the step
            _logger.warning("QAT checkpoint in %s not used (%s): starting fresh", folder, exc)
            return 0
        return min(int(meta["epoch"]), self.qat_epochs)


def adopt_mlflow_run(folder: Path) -> None:
    """Point MLflow at the run the first job opened, so the continued job logs to the same run."""
    try:
        run_id = str(json.loads((Path(folder) / META_FILE).read_text()).get("mlflow_run_id") or "")
    except (OSError, ValueError):
        return
    if run_id:
        os.environ[MLFLOW_RUN_ENV] = run_id
