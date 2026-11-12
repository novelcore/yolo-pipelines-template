"""Manager for the model-quantization pipeline step."""

import logging
from typing import Optional

from app.logger import setup_logging
from app.models.quantization import QuantizationParams, QuantizationResult
from app.services.quantization_service import QuantizationService


class Manager:
    def __init__(self) -> None:
        setup_logging()
        self._logger = logging.getLogger(__name__)
        self._service = QuantizationService()

    def run(
        self,
        mode: str,
        source_mlflow_run_id: str,
        dataset_dir: str,
        output_dir: str,
        export_dir: str,
        experiment_name: str,
        fp32_checkpoint_path: Optional[str] = None,
        tflite_path: Optional[str] = None,
        qat_run_id: Optional[str] = None,
        image_size: int = 640,
        calibration_frames: int = 512,
        calibration_seed: int = 42,
        parity_frames: int = 100,
        parity_max_abs_error: float = 0.05,
    ) -> QuantizationResult:
        self._logger.info(
            "Starting model-quantization | mode=%s source_run=%s",
            mode,
            source_mlflow_run_id,
        )
        params = QuantizationParams(
            mode=mode,  # type: ignore[arg-type]
            fp32_checkpoint_path=fp32_checkpoint_path,
            tflite_path=tflite_path,
            qat_run_id=qat_run_id,
            source_mlflow_run_id=source_mlflow_run_id,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            export_dir=export_dir,
            experiment_name=experiment_name,
            image_size=image_size,
            calibration_frames=calibration_frames,
            calibration_seed=calibration_seed,
            parity_frames=parity_frames,
            parity_max_abs_error=parity_max_abs_error,
        )
        return self._service.run(params=params)
