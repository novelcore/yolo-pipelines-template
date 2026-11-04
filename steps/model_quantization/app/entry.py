"""model-quantization entry point (Hera) — runs the REAL vendored kubeline
quantization service.

Runs when quantization-mode != none. Reads the composed params (--params), the
training output (--training-result), and the qat output (--qat-result, present in
qat mode), maps them to the vendored kubeline ``Manager.run()``:
  - ptq: export INT8 TFLite from the FP32 checkpoint + FP32-vs-INT8 parity
  - qat: pass through the qat-produced INT8 TFLite + parity
and emits ``quantization-result``. Platform-injected env (``LAKEFS_*``,
``MLFLOW_TRACKING_URI``) is read by the vendored ``Config()`` automatically.

MLflow auth is not injected yet (kubecore-operator#868); the INT8 export/parity
upload to lakeFS regardless, and the vendored service guards its MLflow calls.
"""

import argparse
import json
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

import yaml

from app.services import lakefs_io

READS = ["experiment", "quantization"]

OUTPUT_DIR = Path("/work/output")
DATASET_DIR = "/work/dataset"
RUNS_DIR = "/work/runs"


def kaos_lib_version() -> str:
    """The installed kaos-lib version (distribution `kaos-lib`, or `kubecore` before its first release)."""
    for dist in ("kaos-lib", "kubecore"):
        try:
            return package_version(dist)
        except PackageNotFoundError:
            pass
    return "unknown"


def _load(arg: str) -> dict:
    return json.loads(arg) if arg and arg.strip() not in ("", "{}", "null") else {}



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
    parser.add_argument("--qat-result", default="")
    args, _ = parser.parse_known_args()

    cfg = yaml.safe_load(args.params)
    exp = cfg["experiment"]
    q = cfg["quantization"]
    platform = cfg["platform"]

    tr = _load(args.training_result)
    qr = _load(args.qat_result)

    mode = q.get("mode", "ptq")

    # qat mode passes through the qat-produced tflite; ptq exports from the FP32 .pt.
    tflite_s3_uri = qr.get("tflite_s3_uri") if mode == "qat" else None
    qat_run_id = qr.get("mlflow_run_id") if mode == "qat" else None

    # The calibration/parity dataset: the commit training pinned, read through the
    # kaos-lib consumer (one download, or the copy staged on HPC). Its ref/commit
    # go into the artifacts' lakeFS commit metadata.
    _ref = str(tr.get("dataset_ref") or "")
    os.environ["KUBECORE_DATASET_REF"] = _ref
    os.environ["KUBECORE_DATASET_COMMIT"] = str(tr.get("dataset_commit") or "")
    os.makedirs(RUNS_DIR, exist_ok=True)
    data_yaml = lakefs_io.materialize(_ref, str(tr.get("dataset_commit") or ""), DATASET_DIR)
    print(f"[dataset] {_ref} @ {tr.get('dataset_commit')} -> {data_yaml}", flush=True)

    _setup_mlflow_auth()

    from app.manager import Manager

    result = Manager().run(
        mode=mode,
        source_mlflow_run_id=(tr.get("mlflow_run_id") or ""),
        dataset_dir=str(data_yaml.parent),
        output_dir=RUNS_DIR,
        output_bucket=_output_bucket(platform),
        experiment_name=exp["name"],
        fp32_checkpoint_path=tr.get("best_checkpoint_s3"),
        tflite_s3_uri=tflite_s3_uri,
        qat_run_id=qat_run_id,
        image_size=int(q.get("image_size", 640)),
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
    (OUTPUT_DIR / "quantization-result.json").write_text(
        json.dumps(payload, default=str, indent=2)
    )
    print(f"[model-quantization] mode={mode} done. "
          f"result={json.dumps(payload, default=str)[:400]}")


if __name__ == "__main__":
    # Durable error capture: this cluster drops completed-pod logs, so write any
    # failure traceback to the step's output param (survives the pod). The PTQ
    # export (onnx2tf) / QAT litert convert can't be validated without running, so
    # a failure here must be diagnosable from the output, not a lost log.
    import traceback
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "quantization-result.json").write_text(
                json.dumps({"error": "model-quantization failed",
                            "traceback": tb[-4000:]}, indent=2)
            )
        except Exception:
            pass
        print("[model-quantization] FAILED:\n" + tb, flush=True)
        raise

# ci: rebuild to publish the real step image (supersede the stub). Port PRs #12/#13.
