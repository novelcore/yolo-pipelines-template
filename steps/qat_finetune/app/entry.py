"""qat-finetune: fine-tune the trained model with quantization-aware training and save it as an INT8 TFLite."""
from kaos_lib import mlflow as tracking
from kaos_lib import step

from app import names
from app.manager import Manager


def main() -> None:
    ctx = step.start()
    train = step.upstream("model-training").values
    q = ctx.params["quantization"]
    qat = ctx.params["train"].get("qat", {})
    runs = step.workdir()
    tracking.mlflow()  # fails here, before any GPU time, when MLflow cannot be reached

    result = Manager().run(
        fp32_checkpoint_path=str(step.input("checkpoints") / train["best_checkpoint"]),
        source_mlflow_run_id=train["mlflow_run_id"],
        dataset_dir=str(step.input("prepared")),
        output_dir=str(runs),
        export_dir=str(step.output("qat")),
        experiment_name=names.experiment(ctx.app, ctx.experiment),
        image_size=int(q.get("image_size", 640)),
        qat_epochs=int(qat.get("epochs", 10)),
        qat_lr=float(qat.get("lr", 1e-4)),
        calibration_frames=int(q.get("calibration_frames", 512)),
        calibration_seed=int(q.get("calibration_seed", 42)),
        parity_frames=int(q.get("parity_frames", 100)),
        parity_max_abs_error=float(q.get("parity_max_abs_error", 0.05)),
    )
    step.result(**result.model_dump())


if __name__ == "__main__":
    main()
