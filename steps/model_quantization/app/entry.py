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
from pathlib import Path

import yaml

READS = ["experiment", "quantization"]

OUTPUT_DIR = Path("/work/output")
DATASET_DIR = "/work/dataset"
RUNS_DIR = "/work/runs"


def _load(arg: str) -> dict:
    return json.loads(arg) if arg and arg.strip() not in ("", "{}", "null") else {}



def _download_prefix(bucket, prefix, dest):
    """Materialise the dataset locally under {dest}: qat/quant read images + data.yaml
    from dataset_dir directly (no S3-streaming path), and this cluster has no shared FS.
    Downloads every object under s3://{bucket}/{prefix} via the lakeFS S3 gateway."""
    import boto3
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("LAKEFS_ENDPOINT"),
        aws_access_key_id=os.environ.get("LAKEFS_ACCESS_KEY"),
        aws_secret_access_key=os.environ.get("LAKEFS_SECRET_KEY"),
    )
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):]
            if not rel or key.endswith("/"):
                continue
            local = os.path.join(dest, rel)
            os.makedirs(os.path.dirname(local) or dest, exist_ok=True)
            s3.download_file(bucket, key, local)
            n += 1
    print(f"[dataset] downloaded {n} objects from s3://{bucket}/{prefix} -> {dest}", flush=True)


def _fix_data_yaml(dataset_dir):
    """The uploaded data.yaml carries an absolute ``path:`` baked in at dataset-
    creation time (the uploader's machine, e.g. /home/.../speedplus_yolo_101), so
    Ultralytics' calibration/parity loaders resolve images to a path that does not
    exist in-pod. Repoint it at the locally-downloaded tree."""
    p = os.path.join(dataset_dir, "data.yaml")
    if not os.path.exists(p):
        return
    with open(p) as f:
        dy = yaml.safe_load(f) or {}
    dy["path"] = dataset_dir
    for split in ("train", "val", "test"):
        if os.path.isdir(os.path.join(dataset_dir, "images", split)):
            dy[split] = f"images/{split}"
    with open(p, "w") as f:
        yaml.safe_dump(dy, f, sort_keys=False)
    print(f"[dataset] repointed {p} -> path={dataset_dir} "
          f"splits={[s for s in ('train','val','test') if s in dy]}", flush=True)


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
    parser.add_argument("--qat-result", default="")
    args, _ = parser.parse_known_args()

    cfg = yaml.safe_load(args.params)
    exp = cfg["experiment"]
    q = cfg["quantization"]
    platform = cfg["platform"]

    tr = _load(args.training_result)
    qr = _load(args.qat_result)

    mode = q.get("mode", "ptq")
    repo = platform["lakefs"]["repository"]

    # qat mode passes through the qat-produced tflite; ptq exports from the FP32 .pt.
    tflite_s3_uri = qr.get("tflite_s3_uri") if mode == "qat" else None
    qat_run_id = qr.get("mlflow_run_id") if mode == "qat" else None

    # Materialise the calibration dataset locally (dirs + download) before the service.
    data = cfg.get("data", {})
    _ref = data.get("ref", "main"); _ver = data.get("version", "") or ""
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    _download_prefix(repo, f"{_ref}/dataset/{_ver}/", DATASET_DIR)
    _fix_data_yaml(DATASET_DIR)

    _setup_mlflow_auth()

    from app.manager import Manager

    result = Manager().run(
        mode=mode,
        source_mlflow_run_id=(tr.get("mlflow_run_id") or ""),
        dataset_dir=DATASET_DIR,
        output_dir=RUNS_DIR,
        output_bucket=repo,
        output_prefix=q.get("output_prefix", "quantization"),
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
