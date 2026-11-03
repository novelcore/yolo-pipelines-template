"""Manager for the model-quantization pipeline step."""

import logging
import os
from typing import Optional


from app.logger import setup_logging
from app.models.config import Config
from app.models.quantization import QuantizationParams, QuantizationResult
from app.services.quantization_service import QuantizationService


class Manager:
    def __init__(self, config: Optional[Config] = None) -> None:
        self._config = config or Config()
        setup_logging(level=self._config.log_level)
        self._logger = logging.getLogger(__name__)

        if self._config.mlflow_tracking_username:
            os.environ["MLFLOW_TRACKING_USERNAME"] = self._config.mlflow_tracking_username
        if self._config.mlflow_tracking_password:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = self._config.mlflow_tracking_password

        self._service = QuantizationService(
            mlflow_tracking_uri=self._config.mlflow_tracking_uri,
        )

    def run(
        self,
        mode: str,
        source_mlflow_run_id: str,
        dataset_dir: str,
        output_dir: str,
        output_bucket: str,
        experiment_name: str,
        fp32_checkpoint_path: Optional[str] = None,
        tflite_s3_uri: Optional[str] = None,
        qat_run_id: Optional[str] = None,
        image_size: int = 640,
        calibration_frames: int = 512,
        calibration_seed: int = 42,
        parity_frames: int = 100,
        parity_max_abs_error: float = 0.05,
    ) -> QuantizationResult:
        self._logger.info(
            "Starting %s | mode=%s source_run=%s",
            self._config.app_name,
            mode,
            source_mlflow_run_id,
        )
        params = QuantizationParams(
            mode=mode,  # type: ignore[arg-type]
            fp32_checkpoint_path=fp32_checkpoint_path,
            tflite_s3_uri=tflite_s3_uri,
            qat_run_id=qat_run_id,
            source_mlflow_run_id=source_mlflow_run_id,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            output_bucket=output_bucket,
            experiment_name=experiment_name,
            image_size=image_size,
            calibration_frames=calibration_frames,
            calibration_seed=calibration_seed,
            parity_frames=parity_frames,
            parity_max_abs_error=parity_max_abs_error,
        )
        return self._service.run(params=params)
