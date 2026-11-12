"""Manager for the QAT fine-tune pipeline step."""

import logging
from typing import Optional

from app.logger import setup_logging
from app.models.quantization import QATParams, QATResult
from app.services.qat_service import QATService


class Manager:
    """Orchestrates the QAT fine-tune step."""

    def __init__(self) -> None:
        setup_logging()
        self._logger = logging.getLogger(__name__)
        self._service = QATService()

    def run(
        self,
        fp32_checkpoint_path: str,
        source_mlflow_run_id: str,
        dataset_dir: str,
        output_dir: str,
        export_dir: str,
        experiment_name: str,
        image_size: int = 640,
        device: Optional[str] = None,
        qat_epochs: int = 10,
        qat_lr: float = 1e-4,
        calibration_frames: int = 512,
        calibration_seed: int = 42,
        parity_frames: int = 100,
        parity_max_abs_error: float = 0.05,
    ) -> QATResult:
        """Execute the QAT fine-tune step."""
        self._logger.info(
            "Starting qat-finetune | source_run=%s checkpoint=%s",
            source_mlflow_run_id,
            fp32_checkpoint_path,
        )

        params = QATParams(
            fp32_checkpoint_path=fp32_checkpoint_path,
            source_mlflow_run_id=source_mlflow_run_id,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            export_dir=export_dir,
            experiment_name=experiment_name,
            image_size=image_size,
            device=device,
            qat_epochs=qat_epochs,
            qat_lr=qat_lr,
            calibration_frames=calibration_frames,
            calibration_seed=calibration_seed,
            parity_frames=parity_frames,
            parity_max_abs_error=parity_max_abs_error,
        )

        return self._service.run(params=params)
