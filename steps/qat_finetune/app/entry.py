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
from pathlib import Path

import yaml

READS = ["experiment", "train", "quantization"]

OUTPUT_DIR = Path("/work/output")
DATASET_DIR = "/work/dataset"
RUNS_DIR = "/work/runs"



def _download_prefix(bucket, prefix, dest):
    """Materialise the dataset locally under {dest}: qat/quant read images + data.yaml
    from dataset_dir directly (no S3-streaming path), and this cluster has no shared FS.
    Downloads every object under s3://{bucket}/{prefix} via the lakeFS S3 gateway."""
    if not (os.environ.get("LAKEFS_ACCESS_KEY") and os.environ.get("LAKEFS_SECRET_KEY")):
        # Off-cluster there is no S3 gateway to fall back to (live job 5154708:
        # a silent fallback ended in boto3 "Unable to locate credentials").
        raise SystemExit(
            f"[qat-finetune] no lakeFS S3 keys and no staged dataset (KUBECORE_DATASET_DIR) — "
            "off-cluster runs need the platform stage-in; in-cluster runs need LAKEFS_ACCESS_KEY/SECRET_KEY")
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


def _staged_dataset_dir():
    """Off-cluster (MeluXina) the platform stages the dataset ref onto the
    node and bind-mounts it read-only at KUBECORE_DATASET_DIR (PRD-1016 F-04).
    Return the directory holding data.yaml — platform standard
    {ref}/dataset/{version}/ first, then a ref whose root IS the dataset — or
    None when nothing is staged (in-cluster: download from the S3 gateway)."""
    root = os.environ.get("KUBECORE_DATASET_DIR")
    if not root or not os.path.isdir(root):
        return None
    version = os.environ.get("KUBECORE_DATASET_VERSION", "")
    for candidate in (os.path.join(root, "dataset", version) if version else None, root):
        if candidate and os.path.exists(os.path.join(candidate, "data.yaml")):
            return candidate
    raise SystemExit(
        f"[qat-finetune] KUBECORE_DATASET_DIR={root} is mounted but holds no data.yaml under "
        "dataset/{version}/ or its root — the staged ref does not carry a dataset in the platform layout.")


def _link_staged(staged, dest):
    """Expose the read-only staged dataset under {dest}: symlink every entry,
    copy data.yaml so _fix_data_yaml can repoint it."""
    import shutil
    for name in os.listdir(staged):
        target = os.path.join(dest, name)
        if os.path.lexists(target):
            continue
        if name == "data.yaml":
            shutil.copyfile(os.path.join(staged, name), target)
        else:
            os.symlink(os.path.join(staged, name), target)
    print(f"[dataset] staged at {staged} -> linked under {dest} (local mode)", flush=True)


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
    args, _ = parser.parse_known_args()

    cfg = yaml.safe_load(args.params)
    exp = cfg["experiment"]
    train = cfg["train"]
    q = cfg["quantization"]
    platform = cfg["platform"]
    qat = train.get("qat", {})

    tr = json.loads(args.training_result) if args.training_result.strip() not in ("", "{}") else {}
    repo = platform["lakefs"]["repository"]

    # Materialise the calibration dataset locally (dirs + download) before the service.
    data = cfg.get("data", {})
    # Platform standard {ref}/dataset/{version}/ — version defaults to the ref
    # (same as dataset-loading / config-validation), never an empty segment.
    _ref = data.get("ref", "main"); _ver = data.get("version", "") or _ref
    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.environ.setdefault("KUBECORE_DATASET_VERSION", _ver)
    staged = _staged_dataset_dir()
    if staged:
        _link_staged(staged, DATASET_DIR)
    else:
        _download_prefix(repo, f"{_ref}/dataset/{_ver}/", DATASET_DIR)
    _fix_data_yaml(DATASET_DIR)

    _setup_mlflow_auth()

    from app.manager import Manager

    result = Manager().run(
        fp32_checkpoint_path=tr.get("best_checkpoint_s3"),
        source_mlflow_run_id=(tr.get("mlflow_run_id") or ""),
        dataset_dir=DATASET_DIR,
        output_dir=RUNS_DIR,
        output_bucket=repo,
        output_prefix=q.get("output_prefix", "quantization"),
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
