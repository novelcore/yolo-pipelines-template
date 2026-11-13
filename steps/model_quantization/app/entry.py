"""model-quantization: make the INT8 TFLite of the trained model (PTQ), or check and save the one qat-finetune made."""
from kaos_lib import mlflow as tracking
from kaos_lib import step

from app import names
from app.manager import Manager


def main() -> None:
    ctx = step.start()
    q = ctx.params["quantization"]
    mode = q.get("mode", "ptq")
    train = step.upstream("model-training").values
    runs = step.workdir()
    # The export writes beside its checkpoint, so the step works on a writable copy of the input.
    checkpoint = step.input("checkpoints", writable=True) / train["best_checkpoint"]
    tflite, qat_run_id = None, None
    if mode == "qat":  # the qat-finetune result is the INT8 model; this step checks and saves it
        qat = step.upstream("qat-finetune")
        tflite, qat_run_id = str(step.input("qat") / qat.values["tflite"]), qat.values["mlflow_run_id"]
    tracking.mlflow()  # fails here when MLflow cannot be reached

    result = Manager().run(
        mode=mode,
        source_mlflow_run_id=train["mlflow_run_id"],
        dataset_dir=str(step.input("prepared")),
        output_dir=str(runs),
        export_dir=str(step.output("quantization")),
        experiment_name=names.experiment(ctx.app, ctx.experiment),
        fp32_checkpoint_path=checkpoint,
        tflite_path=tflite,
        qat_run_id=qat_run_id,
        image_size=int(q.get("image_size", 640)),
        calibration_frames=int(q.get("calibration_frames", 512)),
        calibration_seed=int(q.get("calibration_seed", 42)),
        parity_frames=int(q.get("parity_frames", 100)),
        parity_max_abs_error=float(q.get("parity_max_abs_error", 0.05)),
    )
    step.result(**result.model_dump())


if __name__ == "__main__":
    main()
