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
from pathlib import Path

import yaml

READS = ["experiment", "train", "quantization"]

OUTPUT_DIR = Path("/work/output")
DATASET_DIR = "/work/dataset"
RUNS_DIR = "/work/runs"


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
    main()

# ci: rebuild to publish the real step image (supersede the stub). Port PRs #12/#13.
