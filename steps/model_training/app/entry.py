"""model-training entry point (Hera) — runs the REAL vendored kubeline trainer.

Reads the composed params.yaml (--params), maps its config sections to the
vendored kubeline ``Manager.run()`` keyword args, and runs a full YOLO
pose-training job with MLflow tracking + lakeFS checkpoint upload. The platform
injects ``LAKEFS_ENDPOINT/LAKEFS_ACCESS_KEY/LAKEFS_SECRET_KEY`` and
``MLFLOW_TRACKING_URI`` as env, which the vendored ``Config()`` reads
automatically (field names match).

Dataset: the cluster is manifest-only (no shared FS between step pods), so we use
the kubeline S3-streaming mode — the trainer streams images/labels straight from
the lakeFS S3 gateway at the bucket/prefix dataset-loading resolved
(``--manifest-summary``): ``{commit}/`` of the dataset repository (the layout
sits at the repository root), pinned to the commit the picked
``<dataset-repo>@<version>`` pointed at when the run started.

MLflow auth (Zitadel machine JWT) is NOT injected by the platform yet, so MLflow
logging is made NON-FATAL: with no auth present we disable yolo's built-in
MLflow callback (its only fatal path), leaving training + lakeFS checkpoint upload
fully functional. It re-enables automatically once auth is wired.
"""

import argparse
import json
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from pathlib import Path

import yaml

from app.services import lakefs_io

# Config sections this step consumes (declared in pipeline.py reads=).
READS = ["experiment", "dataset", "data", "model", "train", "image_processing", "logging"]

OUTPUT_DIR = Path("/work/output")   # where the platform collects declared outputs
DATASET_DIR = "/work/dataset"       # streaming cache/root on this pod's volume
RUNS_DIR = "/work/runs"             # yolo runs/ output


def kaos_lib_version() -> str:
    """The installed kaos-lib version (distribution `kaos-lib`, or `kubecore` before its first release)."""
    for dist in ("kaos-lib", "kubecore"):
        try:
            return package_version(dist)
        except PackageNotFoundError:
            pass
    return "unknown"


def _mlflow_auth_present() -> bool:
    return bool(
        os.environ.get("ZITADEL_MACHINE_KEY_FILE")
        or os.environ.get("MLFLOW_TRACKING_USERNAME")
        or os.environ.get("MLFLOW_TRACKING_TOKEN")
    )


def _dataset_location(summary: dict) -> tuple[str, str, str]:
    """(bucket, prefix, commit) of this run's dataset: what dataset-loading
    resolved and pinned (its manifest-summary). Without it there is no pinned
    commit to read, so the step stops rather than guess a location."""
    bucket, prefix = str(summary.get("bucket") or ""), str(summary.get("prefix") or "")
    if not (bucket and prefix):
        raise SystemExit("[model-training] no dataset location: --manifest-summary from "
                         "dataset-loading is empty, so there is no pinned dataset commit to read.")
    return bucket, prefix if prefix.endswith("/") else prefix + "/", str(summary.get("dataset_commit") or "")


def _staged() -> bool:
    """True on HPC: the platform staged the pinned dataset commit onto the node
    (bind-mounted read-only, announced as KUBECORE_STAGED_DIR, PRD-1016 F-04 / PRD-1541).
    There is no S3 gateway off-cluster, so the dataset is read as local files through the
    kaos-lib consumer; in-cluster the trainer streams from the lakeFS S3 gateway."""
    return bool(os.environ.get("KUBECORE_STAGED_DIR"))


def _guard_mlflow() -> bool:
    """Force-disable yolo's MLflow callback (it is fatal on error) when MLflow
    auth is not usable.

    The `zitadel` request-auth provider comes with kaos-lib[mlflow], installed as a
    package so its entry point registers. Without a usable credential the callback
    would 401 and KILL training (it opens the run and is fatal), so it is disabled:
    training + lakeFS checkpoint upload still work.
    """
    if _setup_mlflow_auth():
        return True
    try:
        from ultralytics import settings
        settings.update({"mlflow": False})
        print("[model-training] MLflow logging DISABLED (no usable MLflow auth). "
              "Training + lakeFS proceed.")
    except Exception as exc:  # noqa: BLE001 - best-effort, never fatal
        print(f"[model-training] WARN: could not disable the trainer's built-in MLflow logging: {exc}")
    return False


def _as_csv(value):
    return ",".join(value) if isinstance(value, list) else value


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
    """Record the dataset commit this run trained on in its MLflow run: the data lock
    (kaos#1147 F-18), tags ``kaos_lib.data.<repo>.commit`` and ``data.lock.json``.
    In-cluster the trainer streams from the lakeFS S3 gateway, past kaos-lib, so the
    read is recorded here; a staged read (HPC) went through kaos-lib and is
    recorded already. Non-fatal, like the rest of the MLflow logging."""
    if not (mlflow_run_id and ds_ref and commit):
        print("[model-training] data lock not written: no MLflow run or no pinned dataset commit.")
        return False
    try:
        from kaos_lib import datasets, lock
        ref = datasets.DatasetRef.parse(ds_ref)
        lock.record(ref.repo, ref.version, commit, ref.owner)
        lock.write(mlflow_run_id)
    except Exception as exc:  # noqa: BLE001 - lineage is logged best-effort, never kills a trained run
        print(f"[model-training] WARN: data lock not written: {exc}", flush=True)
        return False
    print(f"[model-training] data lock: {ds_ref} @ {commit} -> MLflow run {mlflow_run_id}", flush=True)
    return True


def main() -> None:
    print(f"kaos-lib {kaos_lib_version()}", flush=True)  # the package this image runs
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True,
                        help="Resolved params.yaml content (from compose-and-validate).")
    parser.add_argument("--data-yaml", default="",
                        help="The dataset's data.yaml (verbatim) emitted by dataset-loading: "
                             "kpt_shape / names / flip_idx, so training matches THIS dataset.")
    parser.add_argument("--manifest-summary", default="",
                        help="dataset-loading's {bucket, prefix, ref, commit}: the pinned "
                             "location every step of this run reads.")
    args, _ = parser.parse_known_args()
    cfg = yaml.safe_load(args.params)

    experiment = cfg["experiment"]
    model = cfg["model"]
    train = cfg["train"]
    aug = cfg["image_processing"]
    platform = cfg["platform"]

    opt = train.get("optimizer", {})
    loss = train.get("loss", {})
    cb = train.get("callbacks", {})
    ckpt = train.get("checkpointing", {})
    export = train.get("export", {})

    try:
        summary = json.loads(args.manifest_summary or "{}") or {}
    except ValueError:
        summary = {}
    bucket, prefix, commit = _dataset_location(summary)
    # What the run picked (`<dataset-repo>@<version>`), as dataset-loading recorded it.
    ds_repo = str(summary.get("repo") or bucket)
    ds_version = str(summary.get("version") or "")
    ds_ref = str(summary.get("ref") or cfg.get("dataset") or "")
    print(f"[model-training] dataset s3://{bucket}/{prefix} (dataset={ds_ref} "
          f"commit={commit or 'unresolved'})", flush=True)

    freeze = train.get("freeze")
    freeze = int(freeze) if freeze not in (None, "", "None") else None

    # The kubeline service validates that dataset_dir exists and streams S3 images
    # into it (labels under dataset_dir/labels). On this manifest-only cluster
    # nothing pre-creates it, so make the working dirs the streaming trainer needs.
    # (No data.yaml needed: the service generates the correct pose default when
    # dataset_dir has none — kpt_shape [11,3], names {0: spacecraft}.)
    os.makedirs(RUNS_DIR, exist_ok=True)
    shared = bool(summary.get("shared"))
    if _staged() or shared:
        # Local mode: the pinned commit is on the node (HPC), or the dataset is another
        # project's, which this project's S3 gateway does not hold: kaos-lib downloads it from the
        # owner's lakeFS into the workspace volume. The kaos-lib consumer uses a staged copy as is
        # (it checks the commit; nothing is downloaded) and returns a data.yaml whose path: points
        # at the local files.
        data_yaml = lakefs_io.materialize(ds_ref, commit, DATASET_DIR)
        dataset_dir, source = str(data_yaml.parent), "local"
        print(f"[model-training] dataset {ds_ref} @ {commit} -> {data_yaml} (local mode{', shared' if shared else ''})", flush=True)
    else:
        dataset_dir, source = DATASET_DIR, "s3"
        os.makedirs(os.path.join(DATASET_DIR, "labels"), exist_ok=True)
        # Honour the dataset's OWN data.yaml (kpt_shape / names / flip_idx), handed
        # over by dataset-loading as --data-yaml. Without it the service falls back
        # to the spacecraft default (kpt_shape [11,3], names {0: spacecraft}), so a
        # dataset with any other keypoint count has EVERY label rejected and
        # yolo dies with "No valid labels found in labels/train.cache"
        # (live gke-dev/prd1124yolo, 4-keypoint toy dataset, 2026-08-27). The
        # service copies this file into its temp dir and fixes `path` itself.
        data_yaml_text = (args.data_yaml or "").strip()
        if data_yaml_text:
            try:
                ds_yaml = yaml.safe_load(data_yaml_text) or {}
            except yaml.YAMLError as exc:  # keep the run alive; the default may still fit
                print(f"[model-training] WARN --data-yaml is not valid YAML ({exc}); using the service default")
                ds_yaml = {}
            if isinstance(ds_yaml, dict) and ds_yaml:
                ds_yaml["path"] = DATASET_DIR
                with open(os.path.join(DATASET_DIR, "data.yaml"), "w") as fh:
                    yaml.safe_dump(ds_yaml, fh, sort_keys=False)
                print(f"[model-training] data.yaml from dataset-loading written to {DATASET_DIR} "
                      f"(kpt_shape={ds_yaml.get('kpt_shape')}, names={ds_yaml.get('names')})")
        # Enable manifest-only S3 streaming so BOTH images and labels stream from lakeFS
        # (no shared FS holds local labels). The service reads this manifest and sets
        # s3_stream_labels=True (kubecore-operator: manifest-only cluster).
        with open(os.path.join(DATASET_DIR, "dataset_manifest.json"), "w") as _mf:
            json.dump({"bucket": bucket, "prefix": prefix,
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
        s3_bucket=bucket,
        s3_prefix=prefix,
        pretrained_weights=(model.get("pretrained_weights") or None),
        device=None,  # yolo auto-selects: GPU on gpu-t4, CPU when nvidia.com/gpu=0
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
        # ---- checkpointing: the app's own outputs repository, `main`, under
        # checkpoints/<experiment>/<run>/ (the platform injects CHECKPOINT_BUCKET =
        # outputs-<app> and CHECKPOINT_PREFIX = main/checkpoints; platform.checkpoints
        # carries the same values) ----
        checkpoint_interval=int(ckpt.get("interval_epochs", 10)),
        checkpoint_bucket=(os.environ.get("CHECKPOINT_BUCKET")
                           or platform["checkpoints"]["bucket"]),
        checkpoint_prefix=(os.environ.get("CHECKPOINT_PREFIX")
                           or platform["checkpoints"]["prefix"]),
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
        dataset_repo=ds_repo,
        dataset_version=ds_version or None,
        lakefs_commit=(commit or None),
    )
    if mlflow_on:
        _write_data_lock(getattr(result, "mlflow_run_id", "") or "", ds_ref, commit)

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
    # The dataset this run trained on, for the steps after it (registration).
    payload.update(dataset_ref=ds_ref, dataset_repo=ds_repo, dataset_version=ds_version,
                   dataset_commit=commit, dataset_bucket=bucket, dataset_prefix=prefix)
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
