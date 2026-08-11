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
from pathlib import Path

import yaml

READS = ["data", "model", "registration"]

OUTPUT_DIR = Path("/work/output")   # durable step output (survives the pod)


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
    """Mint a Zitadel bearer token for MLflow directly — no entry-point plugin, no
    root install. #868 mounts the machine key at ZITADEL_MACHINE_KEY_FILE; we reuse the
    vendored token source and set MLFLOW_TRACKING_TOKEN, bypassing the request_auth
    plugin (which would need a build-breaking root install). Returns True on success."""
    key = os.environ.get("ZITADEL_MACHINE_KEY_FILE")
    if not key or not os.path.exists(key):
        return False
    try:
        from app.mlflow_zitadel_auth import _ZitadelTokenSource
        os.environ["MLFLOW_TRACKING_TOKEN"] = _ZitadelTokenSource().token()
        os.environ.pop("MLFLOW_TRACKING_AUTH", None)  # use bearer token, not the plugin
        print("[mlflow] Zitadel bearer token minted -> MLflow ENABLED.", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"[mlflow] token mint failed: {exc} -> MLflow disabled (non-fatal).", flush=True)
        return False


def main() -> None:
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

    # Prefer a quantized artifact bundle if a quantization step ran.
    exported = dict(tr.get("exported_models") or {})
    if qr.get("exported_models"):
        exported.update(qr["exported_models"])

    sample_size = data.get("sample_size")
    try:
        sample_size = int(sample_size) if sample_size not in (None, "", "None") else None
    except (TypeError, ValueError):
        sample_size = None

    from app.manager import Manager

    Manager().run(
        mlflow_run_id=mlflow_run_id,
        best_checkpoint_path=tr.get("best_checkpoint_s3"),
        registered_model_name=(reg.get("registered_model_name") or None),
        promote_to=(reg.get("promote_to") or None),
        dataset_version=(data.get("version") or None),
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
    })
    print(f"[model-registration] registered run={mlflow_run_id} "
          f"model={reg.get('registered_model_name')}")


if __name__ == "__main__":
    main()

# ci: rebuild to publish the real step image (supersede the stub). Port PRs #12/#13.
