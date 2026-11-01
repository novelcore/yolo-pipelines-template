"""Manager for the QAT fine-tune pipeline step.

Wires together Config and QATService.
"""

import logging
import os
from typing import Optional


from app.logger import setup_logging
from app.models.config import Config
from app.models.quantization import QATParams, QATResult
from app.services.qat_service import QATService


class Manager:
    """Orchestrates the QAT fine-tune step."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()

        setup_logging(level=self._config.log_level)
        self._logger = logging.getLogger(__name__)

        if self._config.mlflow_tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self._config.mlflow_tracking_username
        if self._config.mlflow_tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self._config.mlflow_tracking_password

        self._service = QATService(
            mlflow_tracking_uri=self._config.mlflow_tracking_uri,
        )

    def run(
        self,
        fp32_checkpoint_path: str,
        source_mlflow_run_id: str,
        dataset_dir: str,
        output_dir: str,
        output_bucket: str,
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
            "Starting %s | source_run=%s checkpoint=%s",
            self._config.app_name,
            source_mlflow_run_id,
            fp32_checkpoint_path,
        )

        params = QATParams(
            fp32_checkpoint_path=fp32_checkpoint_path,
            source_mlflow_run_id=source_mlflow_run_id,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            output_bucket=output_bucket,
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
