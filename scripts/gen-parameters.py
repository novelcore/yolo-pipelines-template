#!/usr/bin/env python3
"""Generate docs/parameters.md: every run-form field, from the sources of truth.

  * name, default, dropdown options  <- the config/ tree (config.yaml + group folders)
  * type and bounds                  <- the pydantic models in steps/*/app/models/*.py
  * wording                          <- DESCRIPTIONS below (the only hand-written part)

Usage:  python scripts/gen-parameters.py            # rewrite docs/parameters.md
        python scripts/gen-parameters.py --check    # exit 1 if the page is out of date
Needs:  pydantic>=2 and PyYAML.
"""
import importlib.util
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "parameters.md"


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


TRAIN = load("steps/model_training/app/models/training.py", "m_train").TrainingParams
AUG = load("steps/model_training/app/models/training.py", "m_train").AugmentationParams
QUANT = load("steps/model_quantization/app/models/quantization.py", "m_quant").QuantizationParams
QAT = load("steps/qat_finetune/app/models/quantization.py", "m_qat").QATParams
REG = load("steps/model_registration/app/models/registration.py", "m_reg").RegistrationParams
DSET = load("steps/dataset_loading/app/models/dataset.py", "m_ds").YoloDatasetParams

# form field -> (model, model field). Fields not listed here have no model-side bounds.
MODEL = {
    "train-epochs": (TRAIN, "epochs"), "train-batch_size": (TRAIN, "batch_size"),
    "train-image_size": (TRAIN, "image_size"), "train-lrf": (TRAIN, "lrf"),
    "train-warmup_epochs": (TRAIN, "warmup_epochs"), "train-warmup_momentum": (TRAIN, "warmup_momentum"),
    "train-dropout": (TRAIN, "dropout"), "train-label_smoothing": (TRAIN, "label_smoothing"),
    "train-nbs": (TRAIN, "nbs"), "train-freeze": (TRAIN, "freeze"), "train-close_mosaic": (TRAIN, "close_mosaic"),
    "train-seed": (TRAIN, "seed"), "train-loss-pose": (TRAIN, "pose"), "train-loss-kobj": (TRAIN, "kobj"),
    "train-loss-box": (TRAIN, "box"), "train-loss-cls": (TRAIN, "cls"), "train-loss-dfl": (TRAIN, "dfl"),
    "train-checkpointing-interval_epochs": (TRAIN, "checkpoint_interval"),
    "train-optimizer-name": (TRAIN, "optimizer"), "train-optimizer-lr": (TRAIN, "learning_rate"),
    "train-optimizer-momentum": (TRAIN, "momentum"), "train-optimizer-weight_decay": (TRAIN, "weight_decay"),
    "train-callbacks-patience": (TRAIN, "patience"),
    "train-qat-epochs": (QAT, "qat_epochs"), "train-qat-lr": (QAT, "qat_lr"),
    "quantization-image_size": (QUANT, "image_size"),
    "quantization-calibration_frames": (QUANT, "calibration_frames"),
    "quantization-calibration_seed": (QUANT, "calibration_seed"),
    "quantization-parity_frames": (QUANT, "parity_frames"),
    "quantization-parity_max_abs_error": (QUANT, "parity_max_abs_error"),
    "registration-promote_to": (REG, "promote_to"),
    "data-sample_size": (DSET, "sample_size"), "data-seed": (DSET, "seed"),
}
for _f in AUG.model_fields:
    MODEL[f"image_processing-{_f}"] = (AUG, _f)

D = {  # one line each; what the field does for you
    "dataset": "The data to train on: `<dataset>@<version>` (`dataset-spacecraft@v1`, `@latest`), or `<owner>/<dataset>@<version>` for a dataset shared with you. You must set this: the default is an empty dataset in a new project.",
    "experiment-name": "Names the MLflow experiment and the folder the weights go to. Reuse a name to compare runs.",
    "experiment-description": "Free text stored on the MLflow run.",
    "model": "Dropdown: which model size to train.",
    "model-variant": "The model file behind the dropdown (set by `model`).",
    "model-pretrained_weights": "An `s3://` path to your own weights to start from (weights only, training starts at epoch 0). Empty uses the bundled ones.",
    "data-source": "Where the data comes from. Leave on `lakefs`.",
    "data-path_override": "An `s3://` folder to read instead of the picked dataset. Leave empty.",
    "data-sample_size": "Train on this many image and label pairs per split, to make a quick test. Empty uses all of it.",
    "data-seed": "Random seed for the sampling above.",
    "train-epochs": "Training rounds.",
    "train-batch_size": "Images per step, a positive number.",
    "train-image_size": "Input size in pixels, a multiple of 32.",
    "train-cos_lr": "Cosine learning-rate schedule.",
    "train-lrf": "Final learning rate as a fraction of the start.",
    "train-warmup_epochs": "Epochs of learning-rate warm-up.",
    "train-warmup_momentum": "Momentum during warm-up.",
    "train-dropout": "Dropout rate.",
    "train-label_smoothing": "Label smoothing.",
    "train-nbs": "Nominal batch size used to scale gradients.",
    "train-freeze": "Freeze the first N backbone layers. Empty trains everything.",
    "train-amp": "Automatic mixed precision.",
    "train-close_mosaic": "Switch the mosaic augmentation off for the last N epochs.",
    "train-seed": "Training random seed.",
    "train-deterministic": "Deterministic GPU mode.",
    "train-loss-pose": "Weight of the keypoint loss.", "train-loss-kobj": "Weight of the keypoint-objectness loss.",
    "train-loss-box": "Weight of the box loss.", "train-loss-cls": "Weight of the class loss.",
    "train-loss-dfl": "Weight of the distribution-focal loss.",
    "train-checkpointing-interval_epochs": "Save a checkpoint every N epochs. A retried training resumes from the last one.",
    "train-checkpointing-storage_path": "Where checkpoints go. Leave empty: your outputs repository is used.",
    "train-checkpointing-resume_from": "Empty starts fresh; `auto` continues from the latest checkpoint; or a full `s3://` path.",
    "train-export-enabled": "Also export the trained model (TensorRT or ONNX) after training.",
    "train-export-formats": "Comma-separated: `engine`, `onnx`.", "train-export-precisions": "Comma-separated: `fp16`, `int8`.",
    "train-optimizer": "Dropdown: `sgd`, `adam`, `adamw`.",
    "train-optimizer-name": "Optimizer (set by the dropdown).", "train-optimizer-lr": "Starting learning rate.",
    "train-optimizer-momentum": "Momentum (Adam: beta1).", "train-optimizer-weight_decay": "Weight decay.",
    "train-callbacks": "Dropdown: `early_stopping` or `none`.",
    "train-callbacks-early_stopping_enabled": "Stop when the model stops improving (set by the dropdown).",
    "train-callbacks-patience": "Epochs without improvement before stopping. `0` never stops early.",
    "train-qat": "Dropdown: `default_qat` or `no_qat`.",
    "train-qat-epochs": "Fine-tuning epochs of `qat` mode.", "train-qat-lr": "Learning rate of `qat` mode.",
    "image_processing": "Dropdown: `mosaic_default`, `conservative` (light, for small datasets) or `none`.",
    "quantization-mode": "`none` (default) skips both INT8 steps; `ptq` quantizes the trained model; `qat` fine-tunes for INT8 first, then exports.",
    "quantization-image_size": "Must equal `train-image_size`.",
    "quantization-calibration_frames": "Training images used to calibrate the INT8 model; 100 to 10000, checked before training when `quantization-mode` is `ptq` or `qat`.",
    "quantization-calibration_seed": "Seed for picking those images.",
    "quantization-parity_frames": "Images used to compare the INT8 and the full model.",
    "quantization-parity_max_abs_error": "Largest allowed difference between the INT8 and the full model. Reported as the MLflow metric `parity_passed`; it does not fail the run.",
    "registration-registered_model_name": "MLflow model name. Empty uses `object-pose-yolo`.",
    "registration-promote_to": "Give the registered version this alias: `champion` or `challenger`. Empty gives none.",
    "logging-level": "Log level of the steps.", "logging-log_artifacts": "Upload plots and example images to MLflow.",
    "summary-label": "Text printed by the `data-summary` step.",
}
for _f in AUG.model_fields:
    D.setdefault(f"image_processing-{_f}", f"Augmentation `{_f}`.")
D.update({"image_processing-degrees": "Rotation augmentation in degrees. Keep 0 for pose: keypoints are rotation-sensitive.",
          "image_processing-perspective": "Perspective augmentation. Values above 0.001 displace keypoints badly.",
          "image_processing-mosaic": "Probability of the mosaic augmentation.",
          "image_processing-fliplr": "Probability of a left-right flip (needs `flip_idx` in `data.yaml`)."})

ORDER = ["experiment", "dataset", "model", "data", "train", "image_processing", "quantization",
         "registration", "logging", "summary"]
SECTION = {"experiment": "Run", "dataset": "Dataset", "model": "Model", "data": "Data loading", "train": "Training",
           "image_processing": "Augmentation", "quantization": "INT8 (quantization)",
           "registration": "Registration", "logging": "Logging", "summary": "Summary"}


def leaves(node, prefix):
    for k, v in node.items():
        if isinstance(v, dict):
            yield from leaves(v, f"{prefix}-{k}" if prefix else k)
        else:
            yield (f"{prefix}-{k}" if prefix else k), v


def kind(v):
    return {bool: "bool", int: "int", float: "float"}.get(type(v), "text")


def bounds(name):
    if name == "quantization-mode":
        return "one of none, ptq, qat"
    if name == "registration-promote_to":
        return "empty, champion or challenger"
    if name not in MODEL:
        return ""
    model, fld = MODEL[name]
    out = []
    for m in model.model_fields[fld].metadata:
        for attr, sym in (("ge", ">="), ("gt", ">"), ("le", "<="), ("lt", "<"), ("multiple_of", "multiple of")):
            if getattr(m, attr, None) is not None:
                out.append(f"{sym} {getattr(m, attr)}")
        pat = getattr(m, "pattern", None)
        if pat:
            out.append("one of " + ", ".join(pat.strip("^$()").split("|")))
    return ", ".join(out)


def val(v):
    return "empty" if v in ("", None) else f"`{str(v).lower() if isinstance(v, bool) else v}`"


def build():
    cfg = ROOT / "config"
    defaults = {}
    for item in yaml.safe_load((cfg / "config.yaml").read_text())["defaults"]:
        if isinstance(item, dict):
            (k, v), = item.items()
            defaults[k] = v
    rows = {s: [] for s in ORDER}
    groups = {"model": "model", "image_processing": "image_processing", "train/optimizer": "train-optimizer",
              "train/callbacks": "train-callbacks", "train/qat": "train-qat"}
    for f in sorted(cfg.glob("*.yaml")):
        sec = f.stem
        if sec not in ORDER or sec == "summary":
            continue
        for name, v in leaves(yaml.safe_load(f.read_text()), ""):
            rows[sec].append((name, v, ""))
    for gdir, gname in groups.items():
        opts = sorted(p.stem for p in (cfg / gdir).glob("*.yaml"))
        sec = gdir.split("/")[0]
        rows[sec].append((gname, defaults[gdir], "options: " + ", ".join(f"`{o}`" for o in opts)))
        for name, v in leaves(yaml.safe_load((cfg / gdir / f"{defaults[gdir]}.yaml").read_text()), ""):
            rows[sec].append((f"{gname}-{name}" if gname != "model" else f"model-{name}", v,
                              f"default of `{defaults[gdir]}`; the dropdown swaps it"))
    L = ["<!-- GENERATED by scripts/gen-parameters.py - do not edit; run the script. -->", "# All the run-form fields", "",
         "Every field of the **Run** form, with its type, default and allowed values. "
         "Type, default and options come from the pipeline's `config/` folder; bounds come from the "
         "checks the steps run on the values. A field is a path with dashes: `train.batch_size` is `train-batch_size`. "
         "Fields set by a dropdown change when you pick another option, so their default is that of the option shown.", "",
         "The form also has `<step>-class`, `<step>-cpu`, `<step>-mem` and `<step>-disk` for each step, "
         "described in [Run on Slurm](slurm.md).", "",
         "!!! note \"Checked before training\"",
         "    With `quantization-mode` = `ptq` or `qat`, the first step (`config-validation`) checks that "
         "`quantization-calibration_frames` is from 100 to 10000, so a wrong value fails in seconds.", ""]
    for sec in ORDER:
        if not rows[sec]:
            continue
        L += [f"## {SECTION[sec]}", "", "| field | type | default | allowed | what it does |", "|---|---|---|---|---|"]
        for name, v, extra in rows[sec]:
            drop = extra.startswith("options")
            desc = D.get(name, "")
            note = extra if drop else ""
            t = "dropdown" if drop else kind(v)
            allowed = note if drop else bounds(name)
            if t == "text" and v == "" and name in ("data-sample_size", "train-freeze"):
                t = "int or empty"
            L.append(f"| `{name}` | {t} | {val(v)} | {allowed} | {desc} |")
        L.append("")
    return "\n".join(L).replace("  \n", "\n")


if __name__ == "__main__":
    text = build()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text() != text:
            sys.exit("docs/parameters.md is out of date: run  python scripts/gen-parameters.py")
        print("parameters.md is up to date")
    else:
        OUT.write_text(text)
        print(f"wrote {OUT.relative_to(ROOT)}")
