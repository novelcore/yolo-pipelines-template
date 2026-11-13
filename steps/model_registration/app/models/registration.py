from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RegistrationParams(BaseModel):
    """Parameters for the MLflow model registration operation."""

    mlflow_run_id: str = Field(
        description="MLflow run ID produced by the training step"
    )
    experiment_name: str = Field(description="MLflow experiment name of the run")
    run_name: str = Field(description="The pipeline run's name")

    # Where model-training's results are: the outputs repository, the commit that holds them, the
    # run's folder in it, and the checkpoint file later steps load (best.pt, or last.pt).
    outputs_repo: str = Field(description="The app's outputs repository, outputs-<app>")
    outputs_commit: str = Field(description="Commit of the outputs repository that holds the checkpoints")
    outputs_path: str = Field(description="Folder of the run's checkpoints, ending in /")
    checkpoint: str = Field(description="File name of the checkpoint to register as best")
    registered_model_name: str = Field(
        description="Name under which to register the model in the MLflow model registry"
    )
    promote_to: Optional[Literal["champion", "challenger"]] = Field(
        default=None,
        description=(
            "Registry alias to assign to the best.pt version after registration "
            "(champion or challenger); None/unset assigns no alias"
        ),
    )

    # Lineage tags — all optional; passed through to MLflow model version tags
    dataset_version: Optional[str] = Field(default=None)
    dataset_commit: Optional[str] = Field(default=None)
    dataset_sample_size: Optional[int] = Field(default=None)
    prepared_dataset: Optional[str] = Field(
        default=None, description="The dataset version training read: <repository>@<commit>"
    )
    model_variant: Optional[str] = Field(
        default=None, description="YOLO model variant, e.g. yolov8n-pose.pt"
    )
    best_map50: Optional[float] = Field(
        default=None, description="Best validation mAP50 from training"
    )
    exported_models: Optional[dict[str, str]] = Field(
        default=None,
        description=(
            "Map of export labels to the file's reference <repository>@<commit>/<path>, the form "
            "model.pretrained_weights takes, "
            "e.g. {'engine_fp16': 'outputs-app@<commit>/checkpoints/exp/run/best.engine'}."
        ),
    )

    # What the experiment's Models tab shows for the run
    epochs: int = Field(description="Epochs the training run completed")
    image_size: int = Field(description="Training image size")
    val_metrics: dict[str, float] = Field(default_factory=dict, description="Final validation metrics")


class RegistrationResult(BaseModel):
    """Outcome of a completed model registration."""

    registered_model_name: str
    best_version: int = Field(description="MLflow model version number for best.pt")
    last_version: Optional[int] = Field(
        default=None,
        description="MLflow model version number for last.pt; None when last.pt was skipped",
    )
    registered_at: datetime
    promoted_to: Optional[str] = Field(
        default=None,
        description="Registry alias assigned to the best.pt version, if any",
    )
