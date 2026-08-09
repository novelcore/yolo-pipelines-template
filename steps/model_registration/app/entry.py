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

import yaml

READS = ["data", "model", "registration"]


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
    if not _mlflow_auth_present() or not mlflow_run_id:
        reason = (
            "MLflow auth not wired (no ZITADEL_MACHINE_KEY_FILE / "
            "MLFLOW_TRACKING_USERNAME)"
            if not _mlflow_auth_present()
            else "training emitted no mlflow_run_id (MLflow was disabled upstream)"
        )
        print(f"[model-registration] SKIPPED — {reason}. "
              f"Model checkpoint is in lakeFS; registration will run once MLflow "
              f"auth is injected into Hera steps.")
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
    print(f"[model-registration] registered run={mlflow_run_id} "
          f"model={reg.get('registered_model_name')}")


if __name__ == "__main__":
    main()

# ci: rebuild to publish the real step image (supersede the stub). Port PRs #12/#13.
