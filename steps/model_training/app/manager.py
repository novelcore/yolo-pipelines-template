"""Manager for the Model Training pipeline step.

Wires the TrainingService; ``run`` is the single orchestration entry point called by the step.
"""

import logging
from typing import Optional

from app.logger import setup_logging
from app.models.training import (
    AugmentationParams,
    ExportConfig,
    TrainingParams,
    TrainingResult,
)
from app.services.model_training import TrainingService


class Manager:
    """Orchestrates the model training step."""

    def __init__(self, log_level: str = "INFO") -> None:
        setup_logging(log_level)
        self._logger = logging.getLogger(__name__)
        self._service = TrainingService()

    def run(
        self,
        # Identity
        model_variant: str,
        experiment_name: str,
        run_name: str,
        dataset_dir: str,
        output_dir: str,
        checkpoints_dir: str,
        # Weight init
        pretrained_weights: Optional[str] = None,
        # Device
        device: Optional[str] = None,
        # Core schedule (keyword-only from here; all callers use explicit kwargs)
        *,
        epochs: int,
        batch_size: int,
        image_size: int,
        # LR
        learning_rate: float,
        cos_lr: bool,
        lrf: float,
        # Optimizer
        optimizer: str,
        momentum: float,
        weight_decay: float,
        # Warmup
        warmup_epochs: float,
        warmup_momentum: float,
        # Regularization
        dropout: float,
        label_smoothing: float,
        # Efficiency
        nbs: int,
        freeze: Optional[int],
        amp: bool,
        close_mosaic: int,
        seed: int,
        deterministic: bool,
        # Loss gains
        pose: float,
        kobj: float,
        box: float,
        cls: float,
        dfl: float,
        # Early stopping
        patience: int,
        # Checkpointing
        checkpoint_interval: int,
        # Augmentation
        hsv_h: float,
        hsv_s: float,
        hsv_v: float,
        degrees: float,
        translate: float,
        scale: float,
        shear: float,
        perspective: float,
        flipud: float,
        fliplr: float,
        mosaic: float,
        mixup: float,
        copy_paste: float,
        erasing: float,
        bgr: float,
        # Export / Quantization
        export_enabled: bool = False,
        export_formats: Optional[str] = None,
        export_precisions: Optional[str] = None,
    ) -> TrainingResult:
        """Execute the model training step end-to-end."""
        self._logger.info(
            "Starting model-training | experiment=%s model=%s epochs=%d",
            experiment_name,
            model_variant,
            epochs,
        )

        params = TrainingParams(
            model_variant=model_variant,
            experiment_name=experiment_name,
            run_name=run_name,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            checkpoints_dir=checkpoints_dir,
            pretrained_weights=pretrained_weights,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            image_size=image_size,
            learning_rate=learning_rate,
            cos_lr=cos_lr,
            lrf=lrf,
            optimizer=optimizer,
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_epochs=warmup_epochs,
            warmup_momentum=warmup_momentum,
            dropout=dropout,
            label_smoothing=label_smoothing,
            nbs=nbs,
            freeze=freeze,
            amp=amp,
            close_mosaic=close_mosaic,
            seed=seed,
            deterministic=deterministic,
            pose=pose,
            kobj=kobj,
            box=box,
            cls=cls,
            dfl=dfl,
            patience=patience,
            checkpoint_interval=checkpoint_interval,
            augmentation=AugmentationParams(
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                degrees=degrees,
                translate=translate,
                scale=scale,
                shear=shear,
                perspective=perspective,
                flipud=flipud,
                fliplr=fliplr,
                mosaic=mosaic,
                mixup=mixup,
                copy_paste=copy_paste,
                erasing=erasing,
                bgr=bgr,
            ),
            export=ExportConfig(
                enabled=export_enabled,
                formats=export_formats.split(",") if export_formats else [],
                precisions=export_precisions.split(",") if export_precisions else [],
            ),
        )

        result = self._service.run(params=params)

        self._logger.info(
            "Training finished | run_id=%s mAP50=%.4f mAP50-95=%.4f best=%s exports=%d",
            result.mlflow_run_id,
            result.final_map50,
            result.final_map50_95,
            result.best_checkpoint,
            len(result.exported_models),
        )

        return result
