"""YOLO model training service with yolo built-in MLflow integration.

Responsibilities
----------------
1. Validate that dataset_dir is a YOLO dataset: data.yaml, images/train, images/val.
2. Resolve the starting weights: a custom .pt (a local file or an s3:// path in lakeFS) or the
   variant's COCO weights.
3. Write data.yaml to a temp dir (always deleted).
4. Set MLFLOW_EXPERIMENT_NAME and MLFLOW_RUN so that the yolo built-in MLflow callback handles
   all logging.
5. Register yolo callbacks for per-epoch metric logging, console output and periodic MLflow
   checkpoint artifacts.
6. Call model.train() with the full hyperparameter set. A step run again after a lost machine or
   an HPC time limit continues from the checkpoint folder: kaos-lib hooks Ultralytics, nothing here.
7. Copy best.pt and last.pt (and the exports) into the step's output folder; the step API saves it
   under ``checkpoints/<experiment>/<run>/`` of the app's outputs repository.
8. Evaluate best.pt and last.pt on the test split (reporting only) and log metrics + plots to the
   current MLflow run. Skipped with a warning when the test split is not available.
9. Clean up temp dirs.
"""

import logging
import os
import re
import shutil
import tempfile
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from app.models.training import TrainingParams, TrainingResult


def kaos_lib_version() -> str:
    """The installed kaos-lib version."""
    return package_version("kaos-lib")


def logged_model_name(experiment: str) -> str:
    """``<experiment>-best``: the name of a run's best.pt in the experiment's Models tab."""
    return f"{experiment}-best"


_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_S3_URI_RE = re.compile(r"^s3://([^/]+)/(.+)$")

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# yolo metric keys
_METRIC_PRECISION = "metrics/precision(B)"
_METRIC_RECALL = "metrics/recall(B)"
_METRIC_MAP50 = "metrics/mAP50(B)"
_METRIC_MAP50_95 = "metrics/mAP50-95(B)"


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------


class TrainingError(Exception):
    """Raised when a training run fails unrecoverably."""


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class TrainingService:
    """Orchestrates a YOLO training run with yolo built-in MLflow tracking.

    MLflow logging (params, metrics, artifacts) is handled by the yolo MLflow callback; this
    service sets the environment variables that callback reads, copies the weights into the step's
    output folder and does the console output.
    """

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, params: TrainingParams) -> TrainingResult:
        """Execute the full training pipeline end-to-end.

        Raises
        ------
        TrainingError
            On any unrecoverable failure.
        """
        self._validate_params(params)

        # Defer heavy imports so that unit tests can mock them easily
        from ultralytics import YOLO  # noqa: PLC0415

        Path(params.output_dir).mkdir(parents=True, exist_ok=True)

        # Configure yolo's built-in MLflow callback via env vars. It reads MLFLOW_RUN (not
        # MLFLOW_RUN_NAME) for the run name: the run's name makes each run's MLflow run unique.
        os.environ["MLFLOW_EXPERIMENT_NAME"] = params.experiment_name
        os.environ["MLFLOW_RUN"] = params.run_name

        try:
            result = self._run_training(params, YOLO)
        except Exception as exc:
            raise TrainingError(f"Training failed: {exc}") from exc

        return result

    # ------------------------------------------------------------------
    # Core training pipeline
    # ------------------------------------------------------------------

    def _run_training(
        self,
        params: TrainingParams,
        yolo_cls: Any,
    ) -> TrainingResult:
        """Inner pipeline: weights, data.yaml, callbacks, train(), weights into the output folder."""

        # Validate the dataset structure before starting the run
        self._validate_local_dataset(params)

        with tempfile.TemporaryDirectory(prefix="io-model-training-") as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 1. Resolve model path (download from lakeFS if needed)
            model_path = self._resolve_model_path(params, tmp_path)

            # 2. Write data.yaml into the temp dir
            data_yaml_path = self._write_data_yaml(params, tmp_path)

            # 3. Build the yolo YOLO model object
            model = yolo_cls(str(model_path))

            # 4. Register callbacks
            epoch_metrics: dict[str, float] = {}

            model.add_callback(
                "on_train_batch_end",
                self._make_batch_end_callback(epoch_metrics),
            )
            model.add_callback(
                "on_fit_epoch_end",
                self._make_epoch_end_callback(epoch_metrics),
            )
            model.add_callback(
                "on_train_epoch_end",
                self._make_checkpoint_callback(params),
            )
            model.add_callback(
                "on_train_end",
                self._make_train_end_callback(epoch_metrics),
            )

            # 5. Build train() kwargs
            train_kwargs = self._build_train_kwargs(
                params=params,
                data_yaml_path=str(data_yaml_path),
            )

            # A step lost here is run again by the platform; the checkpoint folder it keeps
            # (kaos_lib.checkpoint) is what training resumes from, via kaos_lib's Ultralytics hook.
            trainer = model.train(**train_kwargs)

            # 6. Determine save directory
            save_dir = Path(model.trainer.save_dir)  # type: ignore[union-attr]

            # 7. Weights (and exports) into the step's output folder
            best_checkpoint = self._collect_weights(params, save_dir)

            # 8. Export quantized / optimized models
            exported: dict[str, str] = {}
            if params.export.enabled:
                exported = self._export_models(
                    model=model,
                    params=params,
                    data_yaml_path=str(data_yaml_path),
                )

            # 9. Post-training evaluation on the test split (reporting only)
            self._evaluate_on_test_set(
                yolo_cls=yolo_cls,
                params=params,
                save_dir=save_dir,
                data_yaml_path=str(data_yaml_path),
                tmp_path=tmp_path,
            )

        # Build result — get MLflow run_id from yolo's run
        mlflow_run_id = self._get_mlflow_run_id()

        self._tag_run(mlflow_run_id)

        final_map50 = float(epoch_metrics.get("val/mAP50", 0.0))
        final_map50_95 = float(epoch_metrics.get("val/mAP50_95", 0.0))

        raw_epoch = getattr(trainer, "epoch", None)
        epochs_completed: int = raw_epoch + 1 if raw_epoch is not None else params.epochs

        return TrainingResult(
            experiment_name=params.experiment_name,
            model_variant=params.model_variant,
            mlflow_run_id=mlflow_run_id,
            best_checkpoint=best_checkpoint,
            epochs_completed=epochs_completed,
            image_size=params.image_size,
            final_map50=final_map50,
            final_map50_95=final_map50_95,
            val_metrics={k: v for k, v in epoch_metrics.items() if k.startswith("val/")},
            exported_models=exported,
        )

    # ------------------------------------------------------------------
    # Callback factories
    # ------------------------------------------------------------------

    @staticmethod
    def _make_batch_end_callback(
        epoch_metrics: dict[str, float],
    ) -> Callable[[Any], None]:
        """Return a callback that captures per-batch training losses."""

        def on_train_batch_end(trainer: Any) -> None:
            try:
                tloss = getattr(trainer, "tloss", None)
                if tloss is None:
                    return

                loss_names = ["box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"]
                try:
                    for idx, loss_name in enumerate(loss_names):
                        if idx < len(tloss):
                            epoch_metrics[f"train/{loss_name}"] = float(
                                tloss[idx].item()
                                if hasattr(tloss[idx], "item")
                                else tloss[idx]
                            )
                except (TypeError, IndexError):
                    return

            except Exception as cb_exc:  # noqa: BLE001
                _logger.warning("on_train_batch_end callback error: %s", cb_exc)

        return on_train_batch_end

    @staticmethod
    def _make_epoch_end_callback(
        epoch_metrics: dict[str, float],
    ) -> Callable[[Any], None]:
        """Return a callback that logs per-epoch validation metrics."""

        def on_fit_epoch_end(trainer: Any) -> None:
            try:
                val_metrics_raw = getattr(trainer, "metrics", {}) or {}
                val_metrics: dict[str, float] = {
                    "val/precision": float(val_metrics_raw.get(_METRIC_PRECISION, 0.0)),
                    "val/recall": float(val_metrics_raw.get(_METRIC_RECALL, 0.0)),
                    "val/mAP50": float(val_metrics_raw.get(_METRIC_MAP50, 0.0)),
                    "val/mAP50_95": float(val_metrics_raw.get(_METRIC_MAP50_95, 0.0)),
                }
                epoch_metrics.update(val_metrics)

                current_epoch: int = getattr(trainer, "epoch", 0) + 1

                try:
                    import mlflow  # noqa: PLC0415

                    if mlflow.active_run() is not None:
                        mlflow.log_metrics(val_metrics, step=current_epoch)
                    else:
                        _logger.debug(
                            "No active MLflow run — skipping epoch %d metric logging",
                            current_epoch,
                        )
                except Exception as log_exc:  # noqa: BLE001
                    _logger.warning(
                        "Failed to log epoch metrics to MLflow: %s", log_exc
                    )

            except Exception as cb_exc:  # noqa: BLE001
                _logger.warning("on_fit_epoch_end callback error: %s", cb_exc)

        return on_fit_epoch_end

    def _make_checkpoint_callback(
        self,
        params: TrainingParams,
    ) -> Callable[[Any], None]:
        """Return a callback that logs ``last.pt`` to MLflow every N epochs.

        Resuming is not done here: the platform keeps the checkpoint folder (``KAOS_CHECKPOINT_DIR``) and
        ``kaos_lib`` hooks Ultralytics to save into it and resume from it.
        """

        def on_train_epoch_end(trainer: Any) -> None:
            try:
                epoch: int = trainer.epoch + 1  # yolo epoch is 0-indexed
                if epoch % params.checkpoint_interval != 0:
                    return
                last_pt = Path(trainer.last)
                if not last_pt.exists():
                    return

                # Log checkpoint as MLflow artifact
                try:
                    import mlflow  # noqa: PLC0415

                    if mlflow.active_run() is not None:
                        mlflow.log_artifact(
                            str(last_pt),
                            artifact_path=f"checkpoints/epoch_{epoch:04d}",
                        )
                        _logger.info(
                            "Logged epoch %d checkpoint to MLflow artifacts", epoch
                        )
                except Exception as mlflow_exc:  # noqa: BLE001
                    _logger.warning(
                        "Failed to log checkpoint to MLflow: %s", mlflow_exc
                    )

            except Exception as cb_exc:  # noqa: BLE001
                _logger.warning("Logging the checkpoint to MLflow failed: %s", cb_exc)

        return on_train_epoch_end

    @staticmethod
    def _make_train_end_callback(
        epoch_metrics: dict[str, float],
    ) -> Callable[[Any], None]:
        """Return a callback that logs a training completion summary."""

        def on_train_end(trainer: Any) -> None:
            _logger.info(
                "Training complete  best mAP50-95=%.4f  saved to %s",
                float(epoch_metrics.get("val/mAP50_95", 0.0)),
                getattr(trainer, "best", "?"),
            )

        return on_train_end

    # ------------------------------------------------------------------
    # MLflow run ID retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _get_mlflow_run_id() -> str:
        """Return the MLflow run ID from the most recent yolo-managed run."""
        try:
            import mlflow  # noqa: PLC0415

            last_run = mlflow.last_active_run()
            if last_run:
                return last_run.info.run_id
            _logger.warning(
                "mlflow.last_active_run() returned None — "
                "the downstream model_registration step will not be able to "
                "link this training run. Check that the trainer's MLflow "
                "callback is enabled."
            )
        except Exception:  # noqa: BLE001
            _logger.warning(
                "Failed to retrieve MLflow run ID — "
                "model_registration will not be able to link this training run."
            )
        return ""

    @staticmethod
    def _tag_run(run_id: str) -> None:
        """Tag the finished MLflow run with ``kaos_lib.version`` (the kaos-lib package this image runs).

        Uses MlflowClient (not the fluent API) so it works on an already-ended run without
        triggering a new ghost run via mlflow.autolog(). Where the results are in lakeFS is tagged
        by model-registration: the commit exists only once this step has ended.
        """
        if not run_id:
            return
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415

            MlflowClient().set_tag(run_id, "kaos_lib.version", kaos_lib_version())
        except Exception:  # noqa: BLE001
            _logger.warning("Failed to tag MLflow run %s with kaos_lib.version", run_id)

    # ------------------------------------------------------------------
    # Post-training export / quantization
    # ------------------------------------------------------------------

    def _export_models(
        self,
        model: Any,
        params: "TrainingParams",
        data_yaml_path: str,
    ) -> dict[str, str]:
        """Export trained model to requested formats/precisions.

        Copies each exported file into the step's output folder and logs it as an MLflow
        artifact. Errors are non-fatal — training has already succeeded.

        Returns
        -------
        dict[str, str]
            Map of ``{format}_{precision}`` labels to file names in the output folder.
        """
        exported: dict[str, str] = {}
        mlflow_run_id = self._get_mlflow_run_id()

        for fmt in params.export.formats:
            for precision in params.export.precisions:
                label = f"{fmt}_{precision}"

                # ONNX does not support INT8 natively via yolo export
                if fmt == "onnx" and precision == "int8":
                    self._logger.warning(
                        "Skipping %s — ONNX INT8 requires a separate "
                        "onnxruntime quantization pass, not supported by "
                        "the yolo export. Use TensorRT for INT8.",
                        label,
                    )
                    continue

                self._logger.info(
                    "Exporting model: format=%s precision=%s", fmt, precision
                )

                try:
                    export_kwargs: dict[str, Any] = {
                        "format": fmt,
                        "imgsz": params.image_size,
                    }
                    if precision == "fp16":
                        export_kwargs["half"] = True
                    elif precision == "int8":
                        export_kwargs["int8"] = True
                        export_kwargs["data"] = data_yaml_path

                    exported_path = Path(model.export(**export_kwargs))

                    if not exported_path.exists():
                        self._logger.warning(
                            "Export produced no file for %s", label
                        )
                        continue

                    shutil.copy(exported_path, Path(params.checkpoints_dir) / exported_path.name)
                    exported[label] = exported_path.name
                    self._logger.info("Exported %s as %s", label, exported_path.name)

                    # Log as MLflow artifact (run is closed, use MlflowClient)
                    self._log_mlflow_artifact(
                        exported_path,
                        run_id=mlflow_run_id,
                        artifact_path="exports",
                    )

                except Exception as exc:  # noqa: BLE001
                    self._logger.error(
                        "Export failed for %s: %s", label, exc
                    )

        return exported

    @staticmethod
    def _log_mlflow_artifact(
        local_path: Path, run_id: str, artifact_path: str
    ) -> None:
        """Log a file as an MLflow artifact via MlflowClient (works on ended runs)."""
        if not run_id:
            return
        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415

            MlflowClient().log_artifact(
                run_id, str(local_path), artifact_path=artifact_path
            )
        except Exception as exc:  # noqa: BLE001
            _logger.warning("Failed to log artifact to MLflow: %s", exc)

    # ------------------------------------------------------------------
    # Post-training test-set evaluation
    # ------------------------------------------------------------------

    def _evaluate_on_test_set(
        self,
        yolo_cls: Any,
        params: TrainingParams,
        save_dir: Path,
        data_yaml_path: str,
        tmp_path: Path,
    ) -> None:
        """Evaluate best.pt and last.pt on the test split and log to MLflow.

        Reporting only — never fails the run.  Skipped with a warning if
        ``images/test/`` is absent or empty.
        """
        if not self._is_test_split_available(params):
            self._logger.warning(
                "Test split not available (images/test/ absent or empty); "
                "skipping post-training evaluation."
            )
            return

        mlflow_run_id = self._get_mlflow_run_id()

        weights_dir = save_dir / "weights"
        checkpoints: list[tuple[str, Path]] = [
            ("best", weights_dir / "best.pt"),
            ("last", weights_dir / "last.pt"),
        ]

        for label, ckpt_path in checkpoints:
            if not ckpt_path.exists():
                self._logger.warning(
                    "Checkpoint %s not found; skipping test evaluation for '%s'.",
                    ckpt_path,
                    label,
                )
                continue

            try:
                self._logger.info(
                    "Evaluating %s on test split (checkpoint=%s)", label, ckpt_path
                )
                eval_model = yolo_cls(str(ckpt_path))
                val_kwargs: dict[str, Any] = {
                    "data": data_yaml_path,
                    "split": "test",
                    "imgsz": params.image_size,
                    "batch": params.batch_size,
                    "device": params.device,
                    "project": str(tmp_path / "test_eval"),
                    "name": label,
                    "plots": True,
                    "save_json": False,
                }
                results = eval_model.val(**val_kwargs)

                self._log_test_eval_results(
                    label=label,
                    results=results,
                    run_id=mlflow_run_id,
                )
            except Exception as exc:  # noqa: BLE001
                self._logger.warning(
                    "Test evaluation for '%s' failed (non-fatal): %s",
                    label,
                    exc,
                )

    def _is_test_split_available(self, params: TrainingParams) -> bool:
        """Return True iff ``images/test/`` holds at least one image."""
        test_dir = Path(params.dataset_dir) / "images" / "test"
        if not test_dir.is_dir():
            return False
        return any(f.suffix.lower() in _IMAGE_EXTENSIONS for f in test_dir.iterdir())

    def _log_test_eval_results(
        self,
        label: str,
        results: Any,
        run_id: str,
    ) -> None:
        """Log test-eval metrics and plot artifacts to the MLflow run."""
        if not run_id:
            self._logger.warning(
                "No MLflow run_id available; skipping test metric logging for '%s'.",
                label,
            )
            return

        results_dict = getattr(results, "results_dict", None) or {}
        metric_map = {
            _METRIC_PRECISION: f"test/{label}/precision",
            _METRIC_RECALL: f"test/{label}/recall",
            _METRIC_MAP50: f"test/{label}/mAP50",
            _METRIC_MAP50_95: f"test/{label}/mAP50_95",
        }

        try:
            from mlflow.tracking import MlflowClient  # noqa: PLC0415

            client = MlflowClient()
            for src_key, mlflow_key in metric_map.items():
                val = results_dict.get(src_key)
                if val is None:
                    continue
                try:
                    client.log_metric(run_id, mlflow_key, float(val))
                except Exception as exc:  # noqa: BLE001
                    self._logger.warning(
                        "Failed to log metric %s: %s", mlflow_key, exc
                    )

            save_dir = getattr(results, "save_dir", None)
            if save_dir is not None:
                save_dir_path = Path(save_dir)
                if save_dir_path.is_dir():
                    artifact_path = f"eval/test/{label}"
                    try:
                        client.log_artifacts(
                            run_id, str(save_dir_path), artifact_path=artifact_path
                        )
                        self._logger.info(
                            "Logged test/%s metrics and %s artifacts to MLflow",
                            label,
                            artifact_path,
                        )
                    except Exception as exc:  # noqa: BLE001
                        self._logger.warning(
                            "Failed to log test eval artifacts for '%s': %s",
                            label,
                            exc,
                        )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning(
                "Failed to log test eval results for '%s': %s", label, exc
            )

    # ------------------------------------------------------------------
    # Parameter validation
    # ------------------------------------------------------------------

    def _validate_params(self, params: TrainingParams) -> None:
        """Raise TrainingError when the dataset folder is missing."""
        dataset_path = Path(params.dataset_dir)
        if not dataset_path.exists():
            raise TrainingError(
                f"--dataset-dir does not exist: {params.dataset_dir}"
            )
        if not dataset_path.is_dir():
            raise TrainingError(
                f"--dataset-dir is not a directory: {params.dataset_dir}"
            )

    # ------------------------------------------------------------------
    # Local dataset validation
    # ------------------------------------------------------------------

    def _validate_local_dataset(self, params: TrainingParams) -> None:
        """Pre-training sanity check for local mode datasets.

        Verifies that data.yaml, images/train/, and images/val/ all exist
        and contain at least one file each. Logs the structure for
        observability. Raises TrainingError on hard failures.
        """
        dataset_path = Path(params.dataset_dir).resolve()

        # data.yaml must exist
        data_yaml = dataset_path / "data.yaml"
        if not data_yaml.exists():
            raise TrainingError(
                f"data.yaml not found in dataset_dir: {dataset_path}. "
                "A local dataset_dir needs the dataset's data.yaml."
            )
        self._logger.info("Local dataset validation | data.yaml found at %s", data_yaml)

        # Check required splits
        for split in ("train", "val"):
            images_dir = dataset_path / "images" / split
            if not images_dir.exists():
                raise TrainingError(
                    f"images/{split}/ directory not found in dataset_dir: {dataset_path}"
                )
            image_files = [
                p for p in images_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise TrainingError(
                    f"images/{split}/ directory is empty (no image files): {images_dir}"
                )
            self._logger.info(
                "Local dataset validation | images/%s: %d files found",
                split,
                len(image_files),
            )

        # Log optional test split presence without raising
        test_dir = dataset_path / "images" / "test"
        if test_dir.exists():
            test_count = sum(
                1 for p in test_dir.iterdir()
                if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            self._logger.info(
                "Local dataset validation | images/test: %d files found", test_count
            )

    # ------------------------------------------------------------------
    # Model path resolution
    # ------------------------------------------------------------------

    def _resolve_model_path(
        self,
        params: TrainingParams,
        tmp_path: Path,
    ) -> Path:
        """Determine the path passed to YOLO().

        Priority (highest first):
        1. pretrained_weights — a local .pt, or ``s3://<repository>/<ref>/<path>`` in lakeFS,
           downloaded to tmp_path; weight-only initialisation.
        2. model_variant — return the bare variant name so that yolo
           downloads COCO pretrained weights from its CDN on first use.
        """
        if params.pretrained_weights:
            return self._fetch_weights(params.pretrained_weights, tmp_path)

        # Bare variant name: yolo downloads the COCO-pretrained asset on
        # first use — INTO THE PATH WE HAND IT. A bare name means the current
        # directory, which is read-only off-cluster (Apptainer SIF; live job
        # 5149025: "[Errno 30] Read-only file system: 'yolov8n-pose.pt'"), so
        # anchor it under the run's writable output dir. The asset is cached
        # there for later runs on the same scratch.
        weights_dir = Path(params.output_dir) / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        return weights_dir / params.model_variant

    def _fetch_weights(self, uri_or_path: str, tmp_path: Path) -> Path:
        """A local .pt as it is; ``s3://<repository>/<ref>/<path>`` downloaded from lakeFS."""
        match = _S3_URI_RE.match(uri_or_path)
        if not match:
            local_pt = Path(uri_or_path)
            if not local_pt.exists():
                raise TrainingError(f"pretrained weights path does not exist: {uri_or_path}")
            return local_pt
        from kaos_lib.lakefs import LakeFS  # noqa: PLC0415

        repo, key = match.group(1), match.group(2)
        ref, _, path = key.partition("/")
        local_pt = tmp_path / Path(path).name
        self._logger.info("Downloading pretrained weights from %s -> %s", uri_or_path, local_pt)
        LakeFS.connect().download_object(repo, ref, path, local_pt)
        return local_pt

    # ------------------------------------------------------------------
    # data.yaml
    # ------------------------------------------------------------------

    def _write_data_yaml(self, params: TrainingParams, tmp_path: Path) -> Path:
        """Write a data.yaml to the temp directory.

        If dataset_dir already contains a data.yaml (every YOLO dataset has one),
        we copy it into the temp dir and update the ``path`` field to the
        absolute dataset_dir. Otherwise we generate a minimal pose-estimation
        template.

        The temp dir is cleaned up by the caller's TemporaryDirectory context
        manager, so data.yaml is always deleted after training.
        """
        dataset_path = Path(params.dataset_dir).resolve()
        source_yaml = dataset_path / "data.yaml"
        dest_yaml = tmp_path / "data.yaml"

        if source_yaml.exists():
            with source_yaml.open() as fh:
                content: dict[str, Any] = yaml.safe_load(fh) or {}
        else:
            _logger.warning(
                "No data.yaml found in dataset_dir=%s; generating a default template.",
                params.dataset_dir,
            )
            content = {
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "kpt_shape": [11, 3],
                "flip_idx": [],
                "names": {0: "spacecraft"},
            }

        # Always set path to the resolved absolute dataset directory
        content["path"] = str(dataset_path)

        with dest_yaml.open("w") as fh:
            yaml.dump(content, fh, default_flow_style=False, sort_keys=False)

        self._logger.debug("Wrote data.yaml to %s", dest_yaml)
        return dest_yaml

    # ------------------------------------------------------------------
    # train() kwargs
    # ------------------------------------------------------------------

    @staticmethod
    def _cap_workers(workers: int) -> int:
        """DataLoader workers within this step's cores, one left for the training process.

        On a shared Slurm node (kubecore-operator PRD-HPC-1609) the step owns one slot's cores,
        but ``os.cpu_count()``, which yolo sizes by, still counts every core of the node.
        kaos-lib's ``runtime.usable_cpus()`` is the slot's cores, else the CPU binding. 0 (single-process
        loading) stays 0.
        """
        if workers <= 0:
            return workers
        from kaos_lib import runtime

        usable = runtime.usable_cpus()
        capped = max(1, min(workers, usable - 1))
        if capped != workers:
            _logger.info("DataLoader workers %d -> %d: this step may use %d cores", workers, capped, usable)
        return capped

    def _resolve_device(self, device: "str | None") -> "str | int | None":
        """Resolve the training device, defaulting to GPU when one is present.

        yolo's ``model.train(device=None)`` does NOT auto-select a GPU — it
        silently falls back to CPU. On a GPU node that means we provision a T4,
        allocate ``nvidia.com/gpu: 1`` to the pod, then train on the CPU anyway
        (observed live: trainer logged ``device=cpu`` despite the GPU being bound).
        So when the caller leaves device unset, pick GPU 0 if CUDA is actually
        available and fall back to CPU otherwise — an explicit choice, logged.
        An explicit ``--device`` (including ``"cpu"``) is always honoured.
        """
        if device is not None and str(device).strip() != "":
            return device
        try:
            import torch

            if torch.cuda.is_available():
                self._logger.info(
                    "device unset and CUDA available — selecting GPU 0 "
                    "(the trainer's device=None would otherwise train on CPU)"
                )
                return "0"
        except Exception as exc:  # torch import / CUDA probe failure → CPU
            self._logger.warning("CUDA probe failed (%s); training on CPU", exc)
        self._logger.info("device unset and no CUDA — training on CPU")
        return "cpu"

    def _build_train_kwargs(
        self,
        params: TrainingParams,
        data_yaml_path: str,
    ) -> dict[str, Any]:
        """Build the keyword arguments dict for model.train().

        A continued step resumes from the checkpoint folder: kaos-lib sets resume itself.
        """
        aug = params.augmentation

        kwargs: dict[str, Any] = {
            "data": data_yaml_path,
            "epochs": params.epochs,
            "device": self._resolve_device(params.device),
            "batch": params.batch_size,
            "imgsz": params.image_size,
            "lr0": params.learning_rate,
            "cos_lr": params.cos_lr,
            "lrf": params.lrf,
            "optimizer": params.optimizer,
            "momentum": params.momentum,
            "weight_decay": params.weight_decay,
            "warmup_epochs": params.warmup_epochs,
            "warmup_momentum": params.warmup_momentum,
            "dropout": params.dropout,
            "label_smoothing": params.label_smoothing,
            "nbs": params.nbs,
            "amp": params.amp,
            "close_mosaic": params.close_mosaic,
            "workers": self._cap_workers(params.workers),
            "seed": params.seed,
            "deterministic": params.deterministic,
            "patience": params.patience,
            # Pose loss gains
            "pose": params.pose,
            "kobj": params.kobj,
            "box": params.box,
            "cls": params.cls,
            "dfl": params.dfl,
            # Output location
            "project": params.output_dir,
            "name": params.experiment_name,
            "plots": True,
            "save": True,
            "save_period": params.checkpoint_interval,
            # Augmentation
            "hsv_h": aug.hsv_h,
            "hsv_s": aug.hsv_s,
            "hsv_v": aug.hsv_v,
            "degrees": aug.degrees,
            "translate": aug.translate,
            "scale": aug.scale,
            "shear": aug.shear,
            "perspective": aug.perspective,
            "flipud": aug.flipud,
            "fliplr": aug.fliplr,
            "mosaic": aug.mosaic,
            "mixup": aug.mixup,
            "copy_paste": aug.copy_paste,
            "erasing": aug.erasing,
            "bgr": aug.bgr,
        }

        if params.freeze is not None:
            kwargs["freeze"] = params.freeze

        return kwargs

    # ------------------------------------------------------------------
    # Weights into the step's output folder
    # ------------------------------------------------------------------

    def _collect_weights(self, params: TrainingParams, save_dir: Path) -> str:
        """Copy best.pt and last.pt into the step's output folder (``checkpoints_dir``).

        Returns
        -------
        str
            The file name of the checkpoint later steps load: best.pt, or last.pt when this run
            produced no best.pt (Ultralytics writes best.pt only when an epoch beats the best
            fitness, so a continued run that peaked before its interruption has none unless
            kaos-lib restored it). Never empty: a run with no weights at all raises.
        """
        best_pt = save_dir / "weights" / "best.pt"
        last_pt = save_dir / "weights" / "last.pt"
        if not best_pt.exists() and not last_pt.exists():
            raise TrainingError(
                f"training finished but left no weights: neither {best_pt} nor {last_pt} exists"
            )
        for weights in (best_pt, last_pt):
            if weights.exists():
                shutil.copy(weights, Path(params.checkpoints_dir) / weights.name)
                self._logger.info("Saved %s in the output folder", weights.name)
        if best_pt.exists():
            return best_pt.name
        self._logger.warning(
            "best.pt not found at %s; later steps get last.pt as the checkpoint.", best_pt
        )
        return last_pt.name
