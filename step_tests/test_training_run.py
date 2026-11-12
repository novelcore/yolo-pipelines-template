"""A real one-epoch training on the CPU, started on the dataset version prepare-data wrote (nothing is downloaded).

A tiny pose dataset, prepare-data keeping 4 images per split, then model-training through its entry: Ultralytics
trains a model built from its own config on those files, and the weights come out as the model `checkpoints`.
Skipped where torch or ultralytics are not installed.
"""
import os
import subprocess
import sys

import pytest
import yaml
from conftest import ROOT, composed_settings, import_in
from kaos_lib import testing

pytest.importorskip("torch", reason="a real run needs torch (CPU is enough)")
ultralytics = pytest.importorskip("ultralytics", reason="a real run needs ultralytics")


def pose_dataset(folder):
    """8 train, 4 val, 4 test images of 64 pixels, one object with two keypoints each."""
    from PIL import Image

    for split, count in (("train", 8), ("val", 4), ("test", 4)):
        (folder / "images" / split).mkdir(parents=True)
        (folder / "labels" / split).mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (64, 64), (i * 25 % 255, 60, 120)).save(folder / "images" / split / f"{split}{i}.jpg")
            (folder / "labels" / split / f"{split}{i}.txt").write_text("0 0.5 0.5 0.4 0.4 0.4 0.5 2 0.6 0.5 2\n")
    (folder / "data.yaml").write_text(yaml.safe_dump({
        "path": ".", "train": "images/train", "val": "images/val", "test": "images/test", "nc": 1,
        "names": {0: "thing"}, "kpt_shape": [2, 3], "flip_idx": [0, 1]}))
    return folder


def test_prepare_data_then_training_runs_one_epoch_on_the_prepared_version(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLO_OFFLINE", "1")
    ultralytics.settings.update({"mlflow": False, "sync": False})  # no MLflow server here
    weights = tmp_path / "start.pt"
    ultralytics.YOLO("yolov8n-pose.yaml").save(weights)  # built from Ultralytics' own config: no download

    prepare = import_in("prepare_data", "app.entry")
    settings = composed_settings(data={"sample_size": "4", "seed": 42})
    with testing.running(tmp_path / "prepare", kind="transform", name="prepare-data", params=settings,
                         inputs={"dataset": pose_dataset(tmp_path / "raw")}, produces=[("dataset", "prepared")]) as first:
        prepare.main()
    prepared = first.output("prepared")
    assert len(list((prepared / "images" / "train").iterdir())) == 4

    train = import_in("model_training", "app.entry")
    monkeypatch.setattr(train.tracking, "mlflow", lambda *a, **k: None)
    settings = composed_settings()
    settings["model"]["pretrained_weights"] = str(weights)
    settings["train"].update(epochs=1, batch_size=2, image_size=64, close_mosaic=0, amp=False)
    with testing.running(tmp_path / "train", kind="train", name="model-training", experiment="toy", params=settings,
                         inputs={"prepared": prepared}, produces=[("model", "checkpoints")]) as second:
        train.main()

    saved = sorted(p.name for p in second.output("checkpoints").iterdir())
    assert "last.pt" in saved
    values = second.result["values"]
    assert values["epochs_completed"] == 1 and values["best_checkpoint"] in saved
    assert values["image_size"] == 64 and values["mlflow_run_id"] == ""  # MLflow was off for this run


def test_a_continued_step_resumes_from_the_checkpoint_folder(tmp_path):
    """Attempt one is asked to stop during its first epoch (exit code 85, "saved, continue me"); attempt two, on an empty
    work folder and the same checkpoint folder, resumes and ends on the second epoch. Nothing in the step's code does it:
    kaos-lib hooks Ultralytics."""
    weights = tmp_path / "start.pt"
    env = {**os.environ, "YOLO_OFFLINE": "1", "KAOS_CHECKPOINT_DIR": str(tmp_path / "checkpoint"), "PYTHONPATH": ""}
    (tmp_path / "checkpoint").mkdir()
    ultralytics.YOLO("yolov8n-pose.yaml").save(weights)
    prepared = pose_dataset(tmp_path / "prepared")

    def attempt(mode):
        return subprocess.run([sys.executable, str(ROOT / "step_tests" / "resume_driver.py"), mode,
                               str(tmp_path / mode), str(prepared), str(weights)],
                              env=env, capture_output=True, text=True, timeout=600)

    first = attempt("first")
    assert first.returncode == 85, first.stderr[-2000:]
    assert (tmp_path / "checkpoint" / "last.pt").is_file() or list((tmp_path / "checkpoint").rglob("last.pt"))
    second = attempt("second")
    assert second.returncode == 0, second.stderr[-2000:]
    assert "kaos-lib: resuming Ultralytics training" in second.stderr
    assert "EPOCHS 2" in second.stdout
