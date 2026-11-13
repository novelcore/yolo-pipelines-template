"""model-training: train YOLO on the dataset `prepare-data` wrote, and save the weights as the model `checkpoints`."""
from kaos_lib import mlflow as tracking
from kaos_lib import step

from app import names
from app.manager import Manager


def _as_csv(value):
    return ",".join(value) if isinstance(value, list) else value


def main() -> None:
    ctx = step.start()
    model, train, aug = ctx.params["model"], ctx.params["train"], ctx.params["image_processing"]
    opt, loss, cb = train.get("optimizer", {}), train.get("loss", {}), train.get("callbacks", {})
    export = train.get("export", {})
    freeze = train.get("freeze")
    freeze = int(freeze) if freeze not in (None, "", "None") else None
    checkpoints = step.output("checkpoints")
    runs = step.workdir()  # Ultralytics' own output
    tracking.mlflow()  # fails here, before any GPU time, when MLflow cannot be reached

    result = Manager(ctx.params["logging"]["level"]).run(
        model_variant=model["variant"],
        experiment_name=names.experiment(ctx.app, ctx.experiment),
        run_name=ctx.run,
        dataset_dir=str(step.input("prepared")),
        output_dir=str(runs),
        checkpoints_dir=str(checkpoints),
        pretrained_weights=(model.get("pretrained_weights") or None),
        device=None,  # yolo auto-selects: GPU when there is one, CPU otherwise
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
        pose=float(loss.get("pose", 12.0)),
        kobj=float(loss.get("kobj", 2.0)),
        box=float(loss.get("box", 7.5)),
        cls=float(loss.get("cls", 0.5)),
        dfl=float(loss.get("dfl", 1.5)),
        patience=int(cb.get("patience", 50)),
        checkpoint_interval=int(train.get("checkpointing", {}).get("interval_epochs", 10)),
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
        export_enabled=bool(export.get("enabled", False)),
        export_formats=_as_csv(export.get("formats")),
        export_precisions=_as_csv(export.get("precisions")),
    )
    step.result(**result.model_dump())


if __name__ == "__main__":
    main()
