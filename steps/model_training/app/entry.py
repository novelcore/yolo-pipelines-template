"""model-training entry point (Hera) — runs the REAL vendored kubeline trainer.

Reads the composed params.yaml (--params), maps its config sections to the
vendored kubeline ``Manager.run()`` keyword args, and runs a full YOLO
pose-training job with MLflow tracking + lakeFS checkpoint upload. The platform
injects ``LAKEFS_ENDPOINT/LAKEFS_ACCESS_KEY/LAKEFS_SECRET_KEY`` and
``MLFLOW_TRACKING_URI`` as env, which the vendored ``Config()`` reads
automatically (field names match).

Dataset: the cluster is manifest-only (no shared FS between step pods), so we use
the kubeline S3-streaming mode — the trainer streams images/labels straight from
the lakeFS S3 gateway using bucket = repo, prefix = ``{ref}/dataset/{version}/``.

MLflow auth (Zitadel machine JWT) is NOT injected by the platform yet, so MLflow
logging is made NON-FATAL: with no auth present we disable Ultralytics' built-in
MLflow callback (its only fatal path), leaving training + lakeFS checkpoint upload
fully functional. It re-enables automatically once auth is wired.
"""

import argparse
import json
import os
from pathlib import Path

import yaml

# Config sections this step consumes (declared in pipeline.py reads=).
READS = ["experiment", "data", "model", "train", "image_processing", "logging"]

OUTPUT_DIR = Path("/work/output")   # where the platform collects declared outputs
DATASET_DIR = "/work/dataset"       # streaming cache/root on this pod's volume
RUNS_DIR = "/work/runs"             # ultralytics runs/ output


def _mlflow_auth_present() -> bool:
    return bool(
        os.environ.get("ZITADEL_MACHINE_KEY_FILE")
        or os.environ.get("MLFLOW_TRACKING_USERNAME")
        or os.environ.get("MLFLOW_TRACKING_TOKEN")
    )


def _staged_dataset_dir() -> str | None:
    """Off-cluster (MeluXina) the platform stages the dataset ref onto the
    node and bind-mounts it read-only at KUBECORE_DATASET_DIR (PRD-1016 F-04).
    Return the directory holding data.yaml for this run — the ref root mirrors
    the whole ref, so the dataset lives under dataset/{version}/ — or None when
    nothing is staged (in-cluster: stream from the lakeFS S3 gateway)."""
    root = os.environ.get("KUBECORE_DATASET_DIR")
    if not root or not os.path.isdir(root):
        return None
    version = os.environ.get("KUBECORE_DATASET_VERSION", "")
    for candidate in (os.path.join(root, "dataset", version) if version else None,
                      os.path.join(root, "dataset", "main"), root):
        if candidate and os.path.exists(os.path.join(candidate, "data.yaml")):
            return candidate
    print(f"[model-training] KUBECORE_DATASET_DIR={root} holds no data.yaml "
          "(looked in dataset/{version}/, dataset/main/, root) — falling back to "
          "S3 streaming.", flush=True)
    return None


def _guard_mlflow() -> bool:
    """Force-disable Ultralytics' MLflow callback (it is fatal on error).

    #868 injects MLflow Zitadel auth, BUT the `mlflow.request_auth_provider`
    'zitadel' plugin only registers via a root package install — which broke the
    image build (reverted in #17). Enabling the callback would therefore 401 and
    KILL training (Ultralytics' MLflow callback opens the run and is fatal). Until
    MLflow auth is wired without the entry-point plugin (bearer-token approach —
    follow-up), disable it: training + lakeFS checkpoint upload still work; MLflow
    logging + registration are a deliberate follow-up.
    """
    if _setup_mlflow_auth():
        return True
    try:
        from ultralytics import settings
        settings.update({"mlflow": False})
        print("[model-training] MLflow logging DISABLED (zitadel auth-plugin not "
              "wired without root-install; follow-up). Training + lakeFS proceed.")
    except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
        print(f"[model-training] WARN: could not disable ultralytics mlflow: {exc}")
    return False


def _as_csv(value):
    return ",".join(value) if isinstance(value, list) else value


def _setup_mlflow_auth() -> bool:
    """Mint a Zitadel bearer token for MLflow directly — no entry-point plugin, no
    root install. #868 mounts the machine key at ZITADEL_MACHINE_KEY_FILE; we reuse the
    vendored token source and set MLFLOW_TRACKING_TOKEN, bypassing the request_auth
    plugin (which would need a build-breaking root install). Returns True on success."""
    if os.environ.get("MLFLOW_TRACKING_TOKEN"):
        # Off-cluster (MeluXina) the submit pod minted the wallet already.
        os.environ.pop("MLFLOW_TRACKING_AUTH", None)
        print("[mlflow] bearer token supplied by the platform -> MLflow ENABLED.", flush=True)
        return True
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
    parser.add_argument("--params", required=True,
                        help="Resolved params.yaml content (from compose-and-validate).")
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(args.params)

    experiment = cfg["experiment"]
    data = cfg["data"]
    model = cfg["model"]
    train = cfg["train"]
    aug = cfg["image_processing"]
    platform = cfg["platform"]

    opt = train.get("optimizer", {})
    loss = train.get("loss", {})
    cb = train.get("callbacks", {})
    ckpt = train.get("checkpointing", {})
    export = train.get("export", {})

    repo = platform["lakefs"]["repository"]
    ref = data.get("ref", "main")
    version = data.get("version", "") or ""

    freeze = train.get("freeze")
    freeze = int(freeze) if freeze not in (None, "", "None") else None

    # The kubeline service validates that dataset_dir exists and streams S3 images
    # into it (labels under dataset_dir/labels). On this manifest-only cluster
    # nothing pre-creates it, so make the working dirs the streaming trainer needs.
    # (No data.yaml needed: the service generates the correct pose default when
    # dataset_dir has none — kpt_shape [11,3], names {0: spacecraft}.)
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.environ.setdefault("KUBECORE_DATASET_VERSION", version or ref)
    staged = _staged_dataset_dir()
    if staged:
        # Local mode: the staged ref is on the node (read-only bind mount).
        dataset_dir, source = staged, "local"
        print(f"[model-training] dataset staged at {staged} -> local mode", flush=True)
    else:
        dataset_dir, source = DATASET_DIR, "s3"
        os.makedirs(os.path.join(DATASET_DIR, "labels"), exist_ok=True)
        # Enable manifest-only S3 streaming so BOTH images and labels stream from lakeFS
        # (no shared FS holds local labels). The service reads this manifest and sets
        # s3_stream_labels=True (kubecore-operator: manifest-only cluster).
        with open(os.path.join(DATASET_DIR, "dataset_manifest.json"), "w") as _mf:
            json.dump({"bucket": repo, "prefix": f"{ref}/dataset/{version}/",
                       "label_keys": {"_present": True}}, _mf)

    mlflow_on = _guard_mlflow()

    # Import AFTER the mlflow guard so the ultralytics setting is honoured.
    from app.manager import Manager

    result = Manager().run(
        # ---- identity ----
        model_variant=model["variant"],
        experiment_name=experiment["name"],
        dataset_dir=dataset_dir,
        output_dir=RUNS_DIR,
        # ---- dataset: staged local dir (HPC) or lakeFS S3 gateway streaming ----
        source=source,
        s3_bucket=repo,
        s3_prefix=f"{ref}/dataset/{version}/",
        pretrained_weights=(model.get("pretrained_weights") or None),
        device=None,  # ultralytics auto-selects: GPU on gpu-t4, CPU when nvidia.com/gpu=0
        # ---- schedule ----
        epochs=int(train["epochs"]),
        batch_size=int(train["batch_size"]),
        image_size=int(train["image_size"]),
        learning_rate=float(opt.get("lr", 0.01)),
        cos_lr=bool(train.get("cos_lr", True)),
        lrf=float(train.get("lrf", 0.01)),
        optimizer=opt.get("name", "SGD"),
        momentum=float(opt.get("momentum", 0.937)),
        weight_decay=float(opt.get("weight_decay", 0.0005)),
        warmup_epochs=float(train.get("warmup_epochs", 3.0)),
        warmup_momentum=float(train.get("warmup_momentum", 0.8)),
        dropout=float(train.get("dropout", 0.0)),
        label_smoothing=float(train.get("label_smoothing", 0.0)),
        nbs=int(train.get("nbs", 64)),
        freeze=freeze,
        amp=bool(train.get("amp", True)),
        close_mosaic=int(train.get("close_mosaic", 10)),
        seed=int(train.get("seed", 0)),
        deterministic=bool(train.get("deterministic", True)),
        # ---- loss gains ----
        pose=float(loss.get("pose", 12.0)),
        kobj=float(loss.get("kobj", 2.0)),
        box=float(loss.get("box", 7.5)),
        cls=float(loss.get("cls", 0.5)),
        dfl=float(loss.get("dfl", 1.5)),
        # ---- early stopping ----
        patience=int(cb.get("patience", 50)),
        # ---- checkpointing (platform injects CHECKPOINT_BUCKET/PREFIX; else derive) ----
        checkpoint_interval=int(ckpt.get("interval_epochs", 10)),
        checkpoint_bucket=os.environ.get("CHECKPOINT_BUCKET", repo),
        checkpoint_prefix=os.environ.get("CHECKPOINT_PREFIX", f"{ref}/checkpoints/"),
        # ---- augmentation ----
        hsv_h=float(aug.get("hsv_h", 0.015)),
        hsv_s=float(aug.get("hsv_s", 0.7)),
        hsv_v=float(aug.get("hsv_v", 0.4)),
        degrees=float(aug.get("degrees", 0.0)),
        translate=float(aug.get("translate", 0.1)),
        scale=float(aug.get("scale", 0.5)),
        shear=float(aug.get("shear", 0.0)),
        perspective=float(aug.get("perspective", 0.0)),
        flipud=float(aug.get("flipud", 0.0)),
        fliplr=float(aug.get("fliplr", 0.0)),
        mosaic=float(aug.get("mosaic", 1.0)),
        mixup=float(aug.get("mixup", 0.0)),
        copy_paste=float(aug.get("copy_paste", 0.0)),
        erasing=float(aug.get("erasing", 0.4)),
        bgr=float(aug.get("bgr", 0.0)),
        # ---- export ----
        export_enabled=bool(export.get("enabled", False)),
        export_formats=_as_csv(export.get("formats")),
        export_precisions=_as_csv(export.get("precisions")),
        # ---- provenance ----
        dataset_version=(version or None),
        lakefs_branch=ref,
    )

    # Emit the declared step output (Hera convention: /work/output/<name>.json).
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        payload = result.model_dump()  # pydantic v2 model
    except AttributeError:
        try:
            from dataclasses import asdict
            payload = asdict(result)
        except Exception:  # noqa: BLE001
            payload = getattr(result, "__dict__", {"result": str(result)})
    payload["_mlflow_logging_enabled"] = mlflow_on
    (OUTPUT_DIR / "training-result.json").write_text(
        json.dumps(payload, default=str, indent=2)
    )
    print(f"[model-training] done. mlflow_logging={mlflow_on} "
          f"result={json.dumps(payload, default=str)[:400]}")


if __name__ == "__main__":
    # Durable error capture: this cluster does not retain completed-pod logs, so
    # write any failure traceback to the step's output param (survives the pod).
    import traceback
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        try:
            OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUTPUT_DIR / "training-result.json").write_text(
                json.dumps({"error": "model-training failed",
                            "traceback": tb[-4000:]}, indent=2)
            )
        except Exception:
            pass
        print("[model-training] FAILED:\n" + tb, flush=True)
        raise
