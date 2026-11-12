"""Domain models for the YOLO model training step."""

from typing import ClassVar, Optional

from pydantic import BaseModel, Field, field_validator


class AugmentationParams(BaseModel):
    """yolo augmentation hyperparameters."""

    hsv_h: float = Field(default=0.015, ge=0.0, le=1.0)
    hsv_s: float = Field(default=0.7, ge=0.0, le=1.0)
    hsv_v: float = Field(default=0.4, ge=0.0, le=1.0)
    degrees: float = Field(default=0.0, ge=0.0, le=360.0)
    translate: float = Field(default=0.1, ge=0.0)
    scale: float = Field(default=0.5, ge=0.0)
    shear: float = Field(default=0.0, ge=0.0)
    perspective: float = Field(default=0.0, ge=0.0, le=0.001)
    flipud: float = Field(default=0.0, ge=0.0, le=1.0)
    fliplr: float = Field(default=0.0, ge=0.0, le=1.0)
    mosaic: float = Field(default=1.0, ge=0.0, le=1.0)
    mixup: float = Field(default=0.0, ge=0.0, le=1.0)
    copy_paste: float = Field(default=0.0, ge=0.0, le=1.0)
    erasing: float = Field(default=0.4, ge=0.0, le=0.9)
    bgr: float = Field(default=0.0, ge=0.0, le=1.0)


class ExportConfig(BaseModel):
    """Post-training export and quantization configuration."""

    enabled: bool = Field(default=False, description="Enable model export after training.")
    formats: list[str] = Field(
        default_factory=list,
        description="Export formats: 'engine' (TensorRT), 'onnx'.",
    )
    precisions: list[str] = Field(
        default_factory=list,
        description="Precision levels: 'fp16', 'int8'.",
    )

    _VALID_FORMATS: ClassVar[set[str]] = {"engine", "onnx"}
    _VALID_PRECISIONS: ClassVar[set[str]] = {"fp16", "int8"}

    @field_validator("formats")
    @classmethod
    def formats_must_be_valid(cls, v: list[str]) -> list[str]:
        invalid = set(v) - cls._VALID_FORMATS
        if invalid:
            raise ValueError(
                f"Invalid export format(s): {invalid}. Valid: {cls._VALID_FORMATS}"
            )
        return v

    @field_validator("precisions")
    @classmethod
    def precisions_must_be_valid(cls, v: list[str]) -> list[str]:
        invalid = set(v) - cls._VALID_PRECISIONS
        if invalid:
            raise ValueError(
                f"Invalid precision(s): {invalid}. Valid: {cls._VALID_PRECISIONS}"
            )
        return v


class TrainingParams(BaseModel):
    """All parameters required to execute a YOLO training run."""

    # ---- Identity ----
    model_variant: str = Field(
        description="YOLO model variant file, e.g. 'yolov8n-pose.pt'."
    )
    experiment_name: str = Field(
        description="MLflow experiment name."
    )
    dataset_dir: str = Field(
        description="Local directory containing the downloaded YOLO dataset."
    )
    output_dir: str = Field(
        description="Local directory for yolo runs/ output."
    )
    run_name: str = Field(
        description="The run's name: the MLflow run's name and the folder of its results."
    )
    checkpoints_dir: str = Field(
        description="The step's output folder: best.pt, last.pt and the exports are copied here."
    )

    # ---- Weight initialisation ----
    pretrained_weights: Optional[str] = Field(
        default=None,
        description=(
            "Local path or s3://<repository>/<ref>/<path> in lakeFS of a custom .pt file. "
            "Loads weights only; training starts from epoch 0."
        ),
    )

    # ---- Core schedule ----
    epochs: int = Field(default=100, gt=0)
    batch_size: int = Field(default=16, gt=0)
    image_size: int = Field(default=640, gt=0, multiple_of=32)

    # ---- Learning rate ----
    learning_rate: float = Field(default=0.01, gt=0.0)
    cos_lr: bool = Field(default=True)
    lrf: float = Field(default=0.01, gt=0.0, le=1.0)

    # ---- Optimizer ----
    optimizer: str = Field(default="SGD", pattern=r"^(SGD|Adam|AdamW|auto)$")
    momentum: float = Field(default=0.937, ge=0.0, lt=1.0)
    weight_decay: float = Field(default=0.0005, ge=0.0)

    # ---- Warmup ----
    warmup_epochs: float = Field(default=3.0, ge=0.0)
    warmup_momentum: float = Field(default=0.8, ge=0.0, lt=1.0)

    # ---- Regularization ----
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    label_smoothing: float = Field(default=0.0, ge=0.0, lt=1.0)

    # ---- Gradient accumulation ----
    nbs: int = Field(default=64, gt=0)

    # ---- Fine-tuning control ----
    freeze: Optional[int] = Field(default=None, ge=0)

    # ---- Training efficiency ----
    amp: bool = Field(default=True)
    close_mosaic: int = Field(default=10, ge=0)
    workers: int = Field(
        default=0,
        ge=0,
        description=(
            "DataLoader worker processes passed to model.train(). 0 = single-process "
            "loading, which structurally avoids the yolo persistent_workers / "
            "close_mosaic deadlock that intermittently hangs small-dataset runs at the "
            "mosaic-close epoch (GPU 0%, no error). Raise for large datasets where "
            "parallel prefetch pays off."
        ),
    )
    seed: int = Field(default=0, ge=0)
    deterministic: bool = Field(default=True)

    # ---- Pose-estimation loss gains ----
    pose: float = Field(default=12.0, gt=0.0)
    kobj: float = Field(default=2.0, gt=0.0)
    box: float = Field(default=7.5, gt=0.0)
    cls: float = Field(default=0.5, gt=0.0)
    dfl: float = Field(default=1.5, gt=0.0)

    # ---- Early stopping ----
    # 0 disables early stopping: yolo treats it as "never stop"
    # (EarlyStopping: `patience or float("inf")`), and it is exactly what
    # config/train/callbacks/none.yaml selects. gt=0 made train.callbacks=none
    # crash every training run before it started.
    patience: int = Field(default=50, ge=0)

    # ---- Training device ----
    device: Optional[str] = Field(
        default=None,
        description=(
            "Device to train on: '0', '0,1', 'cpu', or None for yolo "
            "auto-select. When a single integer GPU index is given (e.g. '0') "
            "that GPU is used."
        ),
    )

    # ---- Checkpointing ----
    checkpoint_interval: int = Field(default=10, gt=0)

    # ---- Augmentation ----
    augmentation: AugmentationParams = Field(default_factory=AugmentationParams)

    # ---- Export / Quantization ----
    export: ExportConfig = Field(default_factory=ExportConfig)


class TrainingResult(BaseModel):
    """Outcome of a completed training run: the values the step hands on."""

    experiment_name: str
    model_variant: str
    mlflow_run_id: str
    best_checkpoint: str = Field(
        description="File name, in the step's output folder, of the checkpoint later steps load: best.pt, or last.pt."
    )
    epochs_completed: int
    image_size: int
    final_map50: float
    final_map50_95: float
    val_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="The last epoch's validation metrics (val/precision, val/recall, val/mAP50, val/mAP50_95).",
    )
    exported_models: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Map of export label to file name in the output folder, e.g. {'engine_fp16': 'best.engine'}. "
            "Empty when export is disabled."
        ),
    )
