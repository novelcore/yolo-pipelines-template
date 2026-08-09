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
from pathlib import Path

import yaml

READS = ["experiment", "quantization"]

OUTPUT_DIR = Path("/work/output")
DATASET_DIR = "/work/dataset"
RUNS_DIR = "/work/runs"


def _load(arg: str) -> dict:
    return json.loads(arg) if arg and arg.strip() not in ("", "{}", "null") else {}


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
    main()

# ci: rebuild to publish the real step image (supersede the stub). Port PRs #12/#13.
