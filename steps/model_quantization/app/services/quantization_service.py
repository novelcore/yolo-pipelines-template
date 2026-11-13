"""Model quantization service.

PTQ path:
    FP32 .pt checkpoint
        → YOLO.export(format='tflite', int8=True, data=<calibration yaml>)
        → INT8 TFLite
        → the step's output folder + MLflow logging

QAT passthrough:
    INT8 TFLite (from qat-finetune)
        → copied into the step's output folder
        → parity test stub (FR-M-03)
        → MLflow logging
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import mlflow
import torch
import yaml
from mlflow.tracking import MlflowClient
from ultralytics import YOLO

from app.models.quantization import ParityReport, QuantizationParams, QuantizationResult
from app.services.parity_test import ParityTestService

# yolo results_dict keys → sanitized MLflow metric suffixes. Covers the
# box (B) detection metrics and, for pose models, the keypoint (P) metrics.
_MAP_METRIC_KEYS: dict[str, str] = {
    "metrics/mAP50(B)": "mAP50B",
    "metrics/mAP50-95(B)": "mAP50-95B",
    "metrics/mAP50(P)": "mAP50P",
    "metrics/mAP50-95(P)": "mAP50-95P",
}


class QuantizationError(Exception):
    """Raised on non-recoverable quantization failures."""


class QuantizationService:
    """Runs PTQ (yolo) or QAT passthrough and logs to MLflow."""

    def __init__(self) -> None:
        self._logger = logging.getLogger(__name__)
        self._parity = ParityTestService()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, params: QuantizationParams) -> QuantizationResult:
        """Dispatch to PTQ or QAT passthrough based on params.mode."""
        mlflow.set_experiment(params.experiment_name)

        if params.mode == "ptq":
            return self._run_ptq(params)
        return self._run_qat_passthrough(params)

    # ------------------------------------------------------------------
    # PTQ path
    # ------------------------------------------------------------------

    def _run_ptq(self, params: QuantizationParams) -> QuantizationResult:
        """PTQ: yolo export(int8=True) → TFLite → output folder → MLflow."""
        assert params.fp32_checkpoint_path is not None

        local_ckpt = self._check_checkpoint(params.fp32_checkpoint_path)
        data_yaml = self._find_data_yaml(params.dataset_dir, params.output_dir)

        with mlflow.start_run(
            tags={"source_run_id": params.source_mlflow_run_id}
        ) as active_run:
            run_id = active_run.info.run_id
            self._logger.info(
                "PTQ run started | run_id=%s checkpoint=%s",
                run_id,
                local_ckpt,
            )

            self._seed_torch(params.calibration_seed)  # FR-M-04
            tflite_path = self._export_ptq(local_ckpt, data_yaml, params)
            tflite = self._save_tflite(tflite_path, params)
            self._log_tflite_artifact(run_id, tflite_path)
            # PTQ exports the FULL model → compare against full-model FP32.
            parity = self._run_parity_and_log(
                run_id=run_id,
                tflite_path=tflite_path,
                fp32_checkpoint_path=local_ckpt,
                params=params,
                headless=False,
            )
            # Task-metric quality signal: INT8 vs FP32 mAP on the val set.
            self._run_map_delta_and_log(
                run_id=run_id,
                fp32_checkpoint_path=local_ckpt,
                tflite_path=tflite_path,
                data_yaml=data_yaml,
                params=params,
            )
            self._log_ptq_run(run_id, params, tflite)

        self._logger.info("PTQ complete | run_id=%s tflite=%s", run_id, tflite)

        return QuantizationResult(
            mlflow_run_id=run_id,
            source_run_id=params.source_mlflow_run_id,
            mode="ptq",
            tflite=tflite,
            parity_passed=parity.parity_passed,
            parity_max_abs_error=parity.max_abs_error,
        )

    def _export_ptq(
        self, checkpoint_path: str, data_yaml: str, params: QuantizationParams
    ) -> str:
        """Run yolo PTQ export and return the local TFLite path."""
        self._logger.info(
            "Exporting PTQ INT8 TFLite | checkpoint=%s data=%s imgsz=%d",
            checkpoint_path,
            data_yaml,
            params.image_size,
        )
        model = YOLO(checkpoint_path)
        exported = model.export(
            format="tflite",
            int8=True,
            data=data_yaml,
            imgsz=params.image_size,
        )
        tflite_path = str(exported)
        self._logger.info("PTQ export complete: %s", tflite_path)
        return tflite_path

    def _log_ptq_run(
        self, run_id: str, params: QuantizationParams, tflite: str
    ) -> None:
        """Log PTQ parameters and the TFLite file name to MLflow."""
        client = MlflowClient()
        items: list[tuple[str, str]] = [
            ("quantization_mode", "ptq"),
            ("quantization_scheme", "per_tensor_int8"),
            ("calibration_frames", str(params.calibration_frames)),
            ("calibration_seed", str(params.calibration_seed)),
            ("image_size", str(params.image_size)),
            ("parity_frames", str(params.parity_frames)),
            ("parity_max_abs_error_threshold", str(params.parity_max_abs_error)),
            ("source_run_id", params.source_mlflow_run_id),
            ("tflite", tflite),
        ]
        for key, value in items:
            try:
                client.log_param(run_id, key, value)
            except Exception as exc:
                self._logger.warning("Failed to log MLflow param %s: %s", key, exc)

    # ------------------------------------------------------------------
    # QAT passthrough
    # ------------------------------------------------------------------

    def _run_qat_passthrough(self, params: QuantizationParams) -> QuantizationResult:
        """QAT passthrough: save qat-finetune's TFLite as this step's output, run parity, log to MLflow."""
        assert params.tflite_path is not None

        tags: dict[str, str] = {"source_run_id": params.source_mlflow_run_id}
        if params.qat_run_id:
            tags["qat_run_id"] = params.qat_run_id

        with mlflow.start_run(tags=tags) as active_run:
            run_id = active_run.info.run_id
            self._logger.info(
                "QAT passthrough run started | run_id=%s tflite=%s",
                run_id,
                params.tflite_path,
            )

            local_tflite = params.tflite_path
            tflite = self._save_tflite(local_tflite, params)
            self._log_tflite_artifact(run_id, local_tflite)
            # The FP32 checkpoint is the headless parity reference. Mirrors PTQ,
            # which checks the checkpoint before parity.
            local_ckpt = (
                self._check_checkpoint(params.fp32_checkpoint_path)
                if params.fp32_checkpoint_path
                else None
            )
            # QAT exports the backbone+neck only (CON-03) → headless FP32.
            parity = self._run_parity_and_log(
                run_id=run_id,
                tflite_path=local_tflite,
                fp32_checkpoint_path=local_ckpt,
                params=params,
                headless=True,
            )
            self._log_qat_passthrough_run(run_id, params, tflite)

        self._logger.info("QAT passthrough complete | run_id=%s", run_id)

        return QuantizationResult(
            mlflow_run_id=run_id,
            source_run_id=params.source_mlflow_run_id,
            mode="qat",
            tflite=tflite,
            parity_passed=parity.parity_passed,
            parity_max_abs_error=parity.max_abs_error,
        )

    def _log_qat_passthrough_run(self, run_id: str, params: QuantizationParams, tflite: str) -> None:
        """Log QAT passthrough parameters to MLflow."""
        client = MlflowClient()
        items: list[tuple[str, str]] = [
            ("quantization_mode", "qat"),
            ("quantization_scheme", "per_tensor_int8"),
            ("calibration_frames", str(params.calibration_frames)),
            ("calibration_seed", str(params.calibration_seed)),
            ("image_size", str(params.image_size)),
            ("parity_frames", str(params.parity_frames)),
            ("parity_max_abs_error_threshold", str(params.parity_max_abs_error)),
            ("source_run_id", params.source_mlflow_run_id),
            ("tflite", tflite),
        ]
        if params.qat_run_id:
            items.append(("qat_run_id", params.qat_run_id))
        for key, value in items:
            try:
                client.log_param(run_id, key, value)
            except Exception as exc:
                self._logger.warning("Failed to log MLflow param %s: %s", key, exc)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _check_checkpoint(self, path: str) -> str:
        """Return the local path of the checkpoint; an empty path is an error."""
        if not path:
            raise QuantizationError(
                "no FP32 checkpoint to quantize: the training result names no checkpoint"
            )
        return path

    def _find_data_yaml(self, dataset_dir: str, work_dir: str) -> str:
        """Locate the YOLO data YAML in the dataset directory and return a copy that points at it.

        Searches for common filenames: data.yaml, dataset.yaml, config.yaml.
        Raises QuantizationError if none found.

        A dataset's ``path`` is whatever its author wrote (often the folder it was made in), not
        where this step has the dataset. Ultralytics reads the calibration and validation images
        from ``path``, so the copy, written to ``work_dir``, has ``path`` set to ``dataset_dir``,
        the way model-training does. The dataset's own file is never changed.
        """
        candidates = ["data.yaml", "dataset.yaml", "config.yaml"]
        for name in candidates:
            candidate = os.path.join(dataset_dir, name)
            if os.path.isfile(candidate):
                self._logger.info("Found dataset YAML: %s", candidate)
                with open(candidate) as fh:
                    content = yaml.safe_load(fh) or {}
                content["path"] = str(Path(dataset_dir).resolve())
                Path(work_dir).mkdir(parents=True, exist_ok=True)
                local = os.path.join(work_dir, "data.yaml")
                with open(local, "w") as fh:
                    yaml.safe_dump(content, fh, sort_keys=False)
                return local
        raise QuantizationError(
            f"No YOLO data YAML found in {dataset_dir!r}. " f"Searched: {candidates}"
        )

    def _save_tflite(self, local_path: str, params) -> str:
        """Copy the INT8 TFLite into the step's output folder; returns its file name.

        The step API saves the folder as ``quantization/<experiment>/<run>/`` of the app's
        outputs repository, for PTQ and for QAT alike.
        """
        shutil.copy(local_path, Path(params.export_dir) / Path(local_path).name)
        self._logger.info("model-quantization: INT8 tflite saved as %s", Path(local_path).name)
        return Path(local_path).name

    def _seed_torch(self, seed: int) -> None:
        """Seed PyTorch RNG before PTQ calibration export (FR-M-04).

        yolo export(int8=True) calls torch ops internally during
        calibration; seeding here gives best-effort reproducibility.
        """
        torch.manual_seed(seed)
        self._logger.info("PyTorch RNG seed fixed | seed=%d", seed)

    def _log_tflite_artifact(self, run_id: str, tflite_path: str) -> None:
        """Log the quantized .tflite to MLflow as a run artifact.

        Non-fatal: the step's output folder is the authoritative artifact store,
        so an MLflow logging failure only warns (consistent with the other
        MLflow logging in this service).
        """
        client = MlflowClient()
        try:
            client.log_artifact(run_id, tflite_path)
            self._logger.info(
                "Logged TFLite artifact to MLflow | run_id=%s file=%s",
                run_id,
                Path(tflite_path).name,
            )
        except Exception as exc:
            self._logger.warning("Failed to log TFLite artifact to MLflow: %s", exc)

    def _run_parity_and_log(
        self,
        run_id: str,
        tflite_path: str,
        fp32_checkpoint_path: Optional[str],
        params: QuantizationParams,
        headless: bool,
    ) -> ParityReport:
        """Run parity test, save report, and log metrics/artifact to MLflow.

        ``headless`` selects the FP32 reference forward: ``False`` for PTQ
        (full-model TFLite) and ``True`` for QAT (backbone+neck TFLite).

        If fp32_checkpoint_path is None (QAT passthrough without a checkpoint),
        parity is skipped and a passing report with zero error is returned.
        """
        if fp32_checkpoint_path is None:
            self._logger.info(
                "Parity test skipped — no FP32 checkpoint provided (QAT without checkpoint)"
            )
            return ParityReport(
                parity_passed=True,
                max_abs_error=0.0,
                threshold=params.parity_max_abs_error,
                frames_tested=0,
            )

        try:
            parity = self._parity.run(
                tflite_path=tflite_path,
                fp32_checkpoint_path=fp32_checkpoint_path,
                dataset_dir=params.dataset_dir,
                image_size=params.image_size,
                parity_frames=params.parity_frames,
                seed=params.calibration_seed,
                max_abs_error_threshold=params.parity_max_abs_error,
                headless=headless,
            )
            report_path: Optional[str] = self._parity.save_report(
                parity, params.output_dir
            )
        except Exception as exc:
            # Parity is DIAGNOSTIC, not a gate. By this point the INT8 model is
            # already exported, copied into the output folder and logged to MLflow, so a
            # parity computation error must not fail the step (which would also lose the
            # saved model). Record a sentinel and continue.
            self._logger.error(
                "Parity test errored (non-fatal; INT8 model already saved): %s",
                exc,
                exc_info=True,
            )
            # Persist the error to MLflow — the pod's scale-from-0 node is torn
            # down after the run, taking its logs with it, so a bare -1.0 sentinel
            # left us blind. A tag survives and pins WHICH error caused the -1.0.
            try:
                MlflowClient().set_tag(
                    run_id, "parity_error", f"{type(exc).__name__}: {exc}"[:490]
                )
            except Exception:  # noqa: BLE001
                pass
            parity = ParityReport(
                parity_passed=False,
                max_abs_error=-1.0,  # sentinel: parity could not be computed
                threshold=params.parity_max_abs_error,
                frames_tested=0,
            )
            report_path = None

        client = MlflowClient()
        try:
            client.log_metric(run_id, "parity_max_abs_error", parity.max_abs_error)
            client.log_metric(run_id, "parity_passed", float(parity.parity_passed))
            if report_path is not None:
                client.log_artifact(run_id, report_path)
        except Exception as exc:
            self._logger.warning("Failed to log parity metrics to MLflow: %s", exc)

        return parity

    # ------------------------------------------------------------------
    # mAP-delta (FP32 vs INT8 task-metric quality)
    # ------------------------------------------------------------------

    def _run_map_delta_and_log(
        self,
        run_id: str,
        fp32_checkpoint_path: str,
        tflite_path: str,
        data_yaml: str,
        params: QuantizationParams,
    ) -> None:
        """Validate FP32 and INT8 on the val set and log the mAP delta.

        This is the *task-level* quality signal parity cannot give: it runs
        ``model.val`` on the FP32 ``.pt`` and the INT8 ``.tflite`` over the same
        val split and logs ``fp32_<k>`` / ``int8_<k>`` / ``delta_<k>`` for box
        (B) and pose (P) mAP. Retention is logged as ``int8_map_retention_mAP50B``
        (int8 ÷ fp32) when the FP32 reference is non-zero.

        Entirely non-fatal — the INT8 artifact is already published. NOTE: this
        needs a *labelled* val split; on a labels-only calibration download (or
        an untrained model) the reference mAP is ~0, in which case the delta is
        uninformative and ``map_delta_reference_map50b`` (logged) will be ~0.
        """
        client = MlflowClient()
        try:
            fp32_metrics = self._validate_map(
                fp32_checkpoint_path, data_yaml, params, tag="fp32"
            )
            int8_metrics = self._validate_map(
                tflite_path, data_yaml, params, tag="int8"
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.error(
                "mAP-delta errored (non-fatal; INT8 artifact already published): %s",
                exc,
                exc_info=True,
            )
            return

        try:
            for raw_key, suffix in _MAP_METRIC_KEYS.items():
                fp32_v = fp32_metrics.get(raw_key)
                int8_v = int8_metrics.get(raw_key)
                if fp32_v is None or int8_v is None:
                    continue
                client.log_metric(run_id, f"fp32_{suffix}", fp32_v)
                client.log_metric(run_id, f"int8_{suffix}", int8_v)
                client.log_metric(run_id, f"delta_{suffix}", fp32_v - int8_v)

            ref = fp32_metrics.get("metrics/mAP50(B)")
            if ref is not None:
                # Self-documenting flag: if ~0 the mAP-delta is not meaningful
                # (unlabelled val or an untrained model).
                client.log_metric(run_id, "map_delta_reference_map50b", ref)
                int8_ref = int8_metrics.get("metrics/mAP50(B)")
                if ref > 1e-6 and int8_ref is not None:
                    client.log_metric(
                        run_id, "int8_map_retention_mAP50B", int8_ref / ref
                    )
            self._logger.info(
                "mAP-delta | fp32_mAP50B=%.5f int8_mAP50B=%.5f fp32_mAP50P=%.5f "
                "int8_mAP50P=%.5f",
                fp32_metrics.get("metrics/mAP50(B)", 0.0),
                int8_metrics.get("metrics/mAP50(B)", 0.0),
                fp32_metrics.get("metrics/mAP50(P)", 0.0),
                int8_metrics.get("metrics/mAP50(P)", 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.warning("Failed to log mAP-delta metrics to MLflow: %s", exc)

    def _validate_map(
        self,
        model_path: str,
        data_yaml: str,
        params: QuantizationParams,
        tag: str,
    ) -> dict[str, float]:
        """Run ``model.val`` and return its ``results_dict`` (box + pose mAP).

        Writes val outputs under ``output_dir`` (the pod's writable volume) and
        disables plots for speed. AutoBackend handles the INT8 ``.tflite``
        dequantization, so the same call works for both ``.pt`` and ``.tflite``.
        """
        self._logger.info("mAP val (%s) | model=%s", tag, model_path)
        model = YOLO(model_path)
        results = model.val(
            data=data_yaml,
            imgsz=params.image_size,
            verbose=False,
            plots=False,
            project=params.output_dir,
            name=f"val_{tag}",
            exist_ok=True,
        )
        results_dict = getattr(results, "results_dict", None) or {}
        return {k: float(v) for k, v in results_dict.items()}
