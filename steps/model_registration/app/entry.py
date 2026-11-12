"""model-registration: register the trained model in the MLflow model registry."""
from kaos_lib import mlflow as tracking
from kaos_lib import step
from kaos_lib.outputs import run_prefix

from app.manager import Manager
from app.models.registration import RegistrationParams


def _write(upstream, name):
    """What `upstream` saved under `name`: its repository and the commit."""
    return next(w for w in upstream.writes if w.name == name)


def main() -> None:
    ctx = step.start()
    reg, data = ctx.params["registration"], ctx.params["data"]
    train = step.upstream("model-training")
    prepared = step.upstream("prepare-data")
    quantized = step.upstream("model-quantization", optional=True)  # None when quantization-mode is none

    def uri(name, kind, file):  # a file of this run's results, pinned to the commit that holds it
        write = _write(train if kind == "checkpoints" else quantized, name)
        return f"s3://{write.repo}/{write.commit}/{run_prefix(ctx.experiment, ctx.run, kind)}{file}"

    exported = {label: uri("checkpoints", "checkpoints", file) for label, file in train.values["exported_models"].items()}
    if quantized:
        exported["int8_tflite"] = uri("quantization", "quantization", quantized.values["tflite"])
    checkpoints, prepared_data, source = _write(train, "checkpoints"), _write(prepared, "prepared"), prepared.reads[0]
    sample_size = data.get("sample_size")
    tracking.mlflow()  # fails here when MLflow cannot be reached

    result = Manager().run(RegistrationParams(
        mlflow_run_id=train.values["mlflow_run_id"],
        experiment_name=ctx.experiment,
        run_name=ctx.run,
        outputs_repo=checkpoints.repo,
        outputs_commit=checkpoints.commit,
        outputs_path=run_prefix(ctx.experiment, ctx.run, "checkpoints"),
        checkpoint=train.values["best_checkpoint"],
        registered_model_name=reg.get("registered_model_name") or "object-pose-yolo",
        promote_to=reg.get("promote_to") or None,
        dataset_version=f"{source.repo}@{source.ref}",  # the dataset the run started from, before prepare-data
        dataset_commit=source.commit,
        dataset_sample_size=int(sample_size) if sample_size not in (None, "", "None") else None,
        prepared_dataset=f"{prepared_data.repo}@{prepared_data.commit}",
        model_variant=train.values["model_variant"],
        best_map50=train.values["final_map50"],
        exported_models=exported or None,
        epochs=train.values["epochs_completed"],
        image_size=train.values["image_size"],
        val_metrics=train.values["val_metrics"],
    ))
    step.result(**result.model_dump(mode="json"))


if __name__ == "__main__":
    main()
