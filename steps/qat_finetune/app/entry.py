"""qat-finetune entry point (Hera) — runs the REAL vendored kubeline QAT service.

Only runs when quantization-mode == qat. Reads the composed params (--params) and
the training step output (--training-result), maps them to the vendored kubeline
``Manager.run()`` (QAT fine-tune of the FP32 checkpoint -> INT8 TFLite, uploaded
to lakeFS, + FP32-vs-INT8 parity), and emits ``qat-result`` for model-quantization
to consume. Platform-injected env (``LAKEFS_*``, ``MLFLOW_TRACKING_URI``) is read
by the vendored ``Config()`` automatically.

Note: QAT logging/linking uses MLflow, whose auth is not injected yet
(kubecore-operator#868); the core INT8 export uploads to lakeFS regardless. The
vendored service guards its MLflow calls. Full QAT-mode behaviour is validated
once MLflow auth lands.
"""

import argparse
import json
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

import yaml

from app.services import lakefs_io

READS = ["experiment", "train", "quantization"]

OUTPUT_DIR = Path("/work/output")
DATASET_DIR = "/work/dataset"
RUNS_DIR = "/work/runs"


def kaos_lib_version() -> str:
    """The installed kaos-lib version (distribution `kaos-lib`)."""
    try:
        return package_version("kaos-lib")
    except PackageNotFoundError:
        return "unknown"


def _output_bucket(platform: dict) -> str:
    """The app's own outputs repository (`outputs-<app>`): the platform injects
    it as CHECKPOINT_BUCKET and mirrors it in platform.checkpoints.bucket."""
    return os.environ.get("CHECKPOINT_BUCKET") or str(
        (platform.get("checkpoints") or {}).get("bucket") or "")


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


def main() -> None:
    print(f"kaos-lib {kaos_lib_version()}", flush=True)  # the package this image runs
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True)
    parser.add_argument("--training-result", default="")
    args, _ = parser.parse_known_args()

    cfg = yaml.safe_load(args.params)
    exp = cfg["experiment"]
    train = cfg["train"]
    q = cfg["quantization"]
    platform = cfg["platform"]
    qat = train.get("qat", {})

    tr = json.loads(args.training_result) if args.training_result.strip() not in ("", "{}") else {}

    # The calibration/parity dataset: the commit training pinned, read through the
    # kaos-lib consumer (one download, or the copy staged on HPC). Its ref/commit
    # go into the artifacts' lakeFS commit metadata.
    _ref = str(tr.get("dataset_ref") or "")
    os.environ["KAOS_DATASET_REF"] = os.environ["KUBECORE_DATASET_REF"] = _ref  # KUBECORE_*: deprecated alias
    os.environ["KAOS_DATASET_COMMIT"] = os.environ["KUBECORE_DATASET_COMMIT"] = str(tr.get("dataset_commit") or "")
    os.makedirs(RUNS_DIR, exist_ok=True)
    data_yaml = lakefs_io.materialize(_ref, str(tr.get("dataset_commit") or ""), DATASET_DIR)
    print(f"[dataset] {_ref} @ {tr.get('dataset_commit')} -> {data_yaml}", flush=True)

    _setup_mlflow_auth()

    from app.manager import Manager

    result = Manager().run(
        fp32_checkpoint_path=tr.get("best_checkpoint_s3"),
        source_mlflow_run_id=(tr.get("mlflow_run_id") or ""),
        dataset_dir=lakefs_io.dataset_root(data_yaml),
        output_dir=RUNS_DIR,
        output_bucket=_output_bucket(platform),
        experiment_name=exp["name"],
        image_size=int(q.get("image_size", 640)),
        qat_epochs=int(qat.get("epochs", 10)),
        qat_lr=float(qat.get("lr", 1e-4)),
        calibration_frames=int(q.get("calibration_frames", 512)),
        calibration_seed=int(q.get("calibration_seed", 42)),
        parity_frames=int(q.get("parity_frames", 100)),
        parity_max_abs_error=float(q.get("parity_max_abs_error", 0.05)),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        payload = result.model_dump()
    except AttributeError:
        payload = getattr(result, "__dict__", {"result": str(result)})
    (OUTPUT_DIR / "qat-result.json").write_text(json.dumps(payload, default=str, indent=2))
    print(f"[qat-finetune] done. result={json.dumps(payload, default=str)[:400]}")


if __name__ == "__main__":
    # Durable error capture: this cluster drops completed-pod logs, so write any
    # failure traceback to the step's output param (survives the pod). The QAT
    # export (torch.export -> litert_torch convert) can't be validated without
    # running, so a failure here must be diagnosable from the output, not a lost log.
    import traceback
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "qat-result.json").write_text(
                json.dumps({"error": "qat-finetune failed",
                            "traceback": tb[-4000:]}, indent=2)
            )
        except Exception:
            pass
        print("[qat-finetune] FAILED:\n" + tb, flush=True)
        raise

# ci: rebuild to publish the real step image (supersede the stub). Port PRs #12/#13.
