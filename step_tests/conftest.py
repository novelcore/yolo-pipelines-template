"""Fixtures for the step tests: a YOLO dataset in a folder, and the steps imported one at a time.

Every step names its package `app`, so a step is imported with `step_app(<dir>)` which swaps
`sys.modules["app"]` for the duration of the `with`. A step is run the way the platform runs it:
`kaos_lib.testing.running` prepares its context, `kaosctl step run` runs it on this machine.
"""
import contextlib
import importlib
import sys
from pathlib import Path

import pytest

try:
    import kaos_lib  # noqa: F401
except ImportError:
    raise pytest.UsageError("step tests need kaos-lib: see step_tests/README.md for the pip install line") from None

ROOT = Path(__file__).resolve().parents[1]

DATA_YAML = "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  0: ship\n"


@contextlib.contextmanager
def step_app(step):
    """Import ``steps/<step>/app`` as ``app`` and restore the previous ``app`` afterwards."""
    saved = {k: v for k, v in sys.modules.items() if k == "app" or k.startswith("app.")}
    for k in saved:
        del sys.modules[k]
    sys.path.insert(0, str(ROOT / "steps" / step))
    try:
        yield
    finally:
        sys.path.remove(str(ROOT / "steps" / step))
        for k in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
            del sys.modules[k]
        sys.modules.update(saved)


def import_in(step, module):
    with step_app(step):
        return importlib.import_module(module)


def make_dataset(folder: Path, train=6, val=3, test=2, size=32) -> Path:
    """A small YOLO dataset: data.yaml, `train`/`val`/`test` images with one label line each, and a README."""
    from PIL import Image

    for split, count in (("train", train), ("val", val), ("test", test)):
        (folder / "images" / split).mkdir(parents=True)
        (folder / "labels" / split).mkdir(parents=True)
        for i in range(count):
            Image.new("RGB", (size, size), (i * 20 % 255, 40, 90)).save(folder / "images" / split / f"{split}{i:02d}.jpg")
            (folder / "labels" / split / f"{split}{i:02d}.txt").write_text("0 0.5 0.5 0.4 0.4\n")
    (folder / "data.yaml").write_text(DATA_YAML)
    (folder / "README.md").write_text("notes\n")
    return folder


@pytest.fixture
def dataset(tmp_path):
    return make_dataset(tmp_path / "raw")


def composed_settings(**overrides) -> dict:
    """The run's settings with every default of config/ (the platform composes them with Hydra; this merges the same
    files). ``overrides`` replace whole sections."""
    import yaml

    def load(rel):
        return yaml.safe_load((ROOT / "config" / rel).read_text())

    train = load("train.yaml")["train"]
    train["optimizer"] = load("train/optimizer/sgd.yaml")
    train["callbacks"] = load("train/callbacks/early_stopping.yaml")
    train["qat"] = load("train/qat/default_qat.yaml")
    settings = {"experiment": load("experiment.yaml")["experiment"], "dataset": load("dataset.yaml")["dataset"],
                "data": load("data.yaml")["data"], "model": load("model/yolov8n.yaml"), "train": train,
                "image_processing": load("image_processing/mosaic_default.yaml"),
                "quantization": load("quantization.yaml")["quantization"],
                "registration": load("registration.yaml")["registration"], "logging": load("logging.yaml")["logging"]}
    settings.update(overrides)
    return settings
