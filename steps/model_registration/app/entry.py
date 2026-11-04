"""model-registration entry point (Hera) — runs the REAL vendored kubeline
registrar.

Reads the composed params.yaml (--params) plus the upstream step outputs
(--training-result, and --quantization-result when a quantization mode ran), maps
them to the vendored kubeline ``Manager.run()``, and registers the trained model
in the MLflow model registry. Platform-injected env (``MLFLOW_TRACKING_URI``,
``LAKEFS_*``) is read automatically by the vendored ``Config()``.

Registration is fundamentally MLflow-based (it needs the training run's
``mlflow_run_id`` and writes to the registry). The platform does not yet inject
MLflow AUTH (Zitadel machine JWT), and training therefore runs with MLflow
disabled and emits no ``mlflow_run_id``. So when auth is absent (or there is no
run id to register against) this step SKIPS with a clear message and exits 0 —
the pipeline still completes end-to-end, and real registration switches on
automatically once MLflow auth is wired.
"""

import argparse
import json
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

import yaml

READS = ["dataset", "data", "model", "registration"]

OUTPUT_DIR = Path("/work/output")   # durable step output (survives the pod)


def kaos_lib_version() -> str:
    """The installed kaos-lib version (distribution `kaos-lib`, or `kubecore` before its first release)."""
    for dist in ("kaos-lib", "kubecore"):
        try:
            return package_version(dist)
        except PackageNotFoundError:
            pass
    return "unknown"


def _write_result(payload: dict) -> None:
    """Write a durable registration-result.json so the step's TRUE outcome
    (registered vs skipped, and why) survives — completed-pod logs are dropped by
    this cluster, so a green 'Succeeded' must carry its own evidence."""
    try:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "registration-result.json").write_text(
            json.dumps(payload, default=str, indent=2)
        )
    except Exception as exc:  # noqa: BLE001 - never fatal
        print(f"[model-registration] WARN: could not write result: {exc}")


def _mlflow_auth_present() -> bool:
    return bool(
        os.environ.get("ZITADEL_MACHINE_KEY_FILE")
        or os.environ.get("MLFLOW_TRACKING_USERNAME")
    )


def _load_json_arg(value: str) -> dict:
    if not value or value.strip() in ("", "{}", "null"):
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {}


def _setup_mlflow_auth() -> bool:
    """MLflow auth through kaos-lib (kaos-lib[mlflow]). ``mlflow()`` sets the tracking URI and
    MLFLOW_TRACKING_AUTH=zitadel; that request-auth provider ships with kaos-lib and takes the
    bearer from one chain: the platform's MLFLOW_TRACKING_TOKEN (HPC), else a token minted
    from the machine key at ZITADEL_MACHINE_KEY_FILE with the org project's audience and minted
    again before it expires. Returns True when MLflow is usable; otherwise MLflow is disabled
    (non-fatal)."""
    try:
        from kaos_lib.mlflow import mlflow
        mlflow()
    except Exception as exc:  # noqa: BLE001 - MLflow is optional for this step
        print(f"[mlflow] {exc} -> MLflow disabled (non-fatal).", flush=True)
        return False
    print("[mlflow] kaos-lib auth (MLFLOW_TRACKING_AUTH=zitadel) -> MLflow ENABLED.", flush=True)
    return True


def _write_data_lock(mlflow_run_id: str, ds_ref: str, commit: str) -> bool:
    """Record the dataset commit the registered model was trained on in its MLflow run:
    the data lock (kaos#1147 F-18), tags ``kaos_lib.data.<repo>.commit`` and
    ``data.lock.json`` (merged with what training wrote). Non-fatal."""
    if not (ds_ref and commit):
        print("[model-registration] data lock not written: training-result has no dataset commit.")
        return False
    try:
        from kaos_lib import datasets, lock
        ref = datasets.DatasetRef.parse(ds_ref)
        lock.record(ref.repo, ref.version, commit, ref.owner)
        lock.write(mlflow_run_id)
    except Exception as exc:  # noqa: BLE001 - lineage is logged best-effort, never blocks registration
        print(f"[model-registration] WARN: data lock not written: {exc}", flush=True)
        return False
    print(f"[model-registration] data lock: {ds_ref} @ {commit} -> MLflow run {mlflow_run_id}", flush=True)
    return True


def main() -> None:
    print(f"kaos-lib {kaos_lib_version()}", flush=True)  # the package this image runs
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--training-result", default="")
    parser.add_argument("--quantization-result", default="")
    args, _ = parser.parse_known_args()

    cfg = yaml.safe_load(args.params)
    data = cfg.get("data", {})
    model = cfg.get("model", {})
    reg = cfg.get("registration", {})

    tr = _load_json_arg(args.training_result)
    qr = _load_json_arg(args.quantization_result)

    mode = str((cfg.get("quantization") or {}).get("mode", "none")).lower()

    # C-06 guard (novelcore/yolo-hera-yolo#30): if a quantization mode was requested,
    # the quant/qat step MUST have produced a successful INT8 artifact. A failed or
    # omitted quant/qat step leaves an empty/error quantization-result; registering the
    # FP32 checkpoint then silently hands back the WRONG (un-quantized) model behind a
    # green check. Fail LOUDLY instead — registration is the terminal step, so failing
    # here also makes the overall workflow report failure (not a false Succeeded).
    if mode != "none":
        quant_ok = bool(qr.get("tflite_s3_uri")) and not qr.get("error")
        if not quant_ok:
            detail = qr.get("error") or "quant/qat step produced no INT8 artifact (no tflite_s3_uri)"
            _write_result({
                "status": "failed",
                "registered": False,
                "reason": f"quantization-mode={mode} requested but not satisfied: {detail}",
                "quantization_mode": mode,
                "best_checkpoint_s3": tr.get("best_checkpoint_s3"),
            })
            raise SystemExit(
                f"[model-registration] FAILED (C-06) — quantization-mode={mode} but the "
                f"quant/qat step did not produce a successful INT8 artifact; refusing to "
                f"register the FP32 model. detail: {str(detail)[:300]}"
            )

    mlflow_run_id = tr.get("mlflow_run_id")

    # Registration needs MLflow. Without auth wired (or a run id to register
    # against), skip non-fatally so the pipeline still completes end-to-end.
    if not _setup_mlflow_auth() or not mlflow_run_id:
        reason = (
            "MLflow auth not wired (no ZITADEL_MACHINE_KEY_FILE / "
            "MLFLOW_TRACKING_USERNAME)"
            if not os.environ.get("ZITADEL_MACHINE_KEY_FILE")
            else "training emitted no mlflow_run_id (MLflow was disabled upstream)"
        )
        print(f"[model-registration] SKIPPED — {reason}. "
              f"Model checkpoint is in lakeFS; registration will run once MLflow "
              f"auth is injected into Hera steps.")
        _write_result({
            "status": "skipped",
            "registered": False,
            "reason": reason,
            "best_checkpoint_s3": tr.get("best_checkpoint_s3"),
            "mlflow_run_id": mlflow_run_id or None,
        })
        return

    # Prefer a quantized artifact bundle if a quantization step ran. Include the
    # INT8 TFLite the quant/qat step produced (quantization-result carries it as
    # tflite_s3_uri) so a successful quantized run registers the INT8 artifact, not
    # only the FP32 checkpoint (C-06).
    exported = dict(tr.get("exported_models") or {})
    if qr.get("exported_models"):
        exported.update(qr["exported_models"])
    if qr.get("tflite_s3_uri"):
        exported["int8_tflite"] = qr["tflite_s3_uri"]

    sample_size = data.get("sample_size")
    try:
        sample_size = int(sample_size) if sample_size not in (None, "", "None") else None
    except (TypeError, ValueError):
        sample_size = None

    _write_data_lock(mlflow_run_id, str(tr.get("dataset_ref") or cfg.get("dataset") or ""),
                     str(tr.get("dataset_commit") or ""))

    from app.manager import Manager

    Manager().run(
        mlflow_run_id=mlflow_run_id,
        best_checkpoint_path=tr.get("best_checkpoint_s3"),
        registered_model_name=(reg.get("registered_model_name") or None),
        promote_to=(reg.get("promote_to") or None),
        # The dataset = `<dataset-repo>@<version>`; the commit pins the exact
        # snapshot model-training read (relayed in training-result).
        dataset_version=(tr.get("dataset_ref") or cfg.get("dataset") or None),
        dataset_commit=(tr.get("dataset_commit") or None),
        dataset_sample_size=sample_size,
        model_variant=(model.get("variant") or tr.get("model_variant")),
        best_map50=tr.get("final_map50"),
        exported_models=(exported or None),
    )
    _write_result({
        "status": "registered",
        "registered": True,
        "mlflow_run_id": mlflow_run_id,
        "registered_model_name": reg.get("registered_model_name"),
        "best_checkpoint_s3": tr.get("best_checkpoint_s3"),
        "best_map50": tr.get("final_map50"),
        "quantization_mode": mode,
        "int8_tflite_s3": qr.get("tflite_s3_uri"),
    })
    print(f"[model-registration] registered run={mlflow_run_id} "
          f"model={reg.get('registered_model_name')}")


if __name__ == "__main__":
    main()

# ci: rebuild to publish the real step image (supersede the stub). Port PRs #12/#13.
