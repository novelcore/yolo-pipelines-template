"""Fixtures: an in-memory lakeFS with a dataset and an outputs repository, and per-step imports.

Every step names its package `app`, so a step is imported with `step_app(<dir>)` which swaps
`sys.modules["app"]` for the duration of the `with`.
"""
import contextlib
import importlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # fake_lakefs.py

try:
    import kaos_lib  # noqa: E402,F401
except ImportError:
    raise pytest.UsageError("step tests need kaos-lib: see step_tests/README.md for the pip install line") from None

from fake_lakefs import FakeLakeFS  # noqa: E402

ENV_VARS = ("LAKEFS_ENDPOINT", "LAKEFS_URL", "LAKEFS_ACCESS_KEY", "LAKEFS_SECRET_KEY", "LAKEFS_BEARER_TOKEN",
            "KAOS_STAGED_DIR", "KAOS_CACHE_DIR", "KAOS_DATASET_CONFIG", "CHECKPOINT_BUCKET",
            "CHECKPOINT_PREFIX", "KAOS_RUN_ID", "KAOS_WORKFLOW_NAME", "KAOS_DATASET_DIR",
            # deprecated aliases, still read by a kaos-lib older than the rename
            "KUBECORE_STAGED_DIR", "KUBECORE_CACHE_DIR", "KUBECORE_DATASET_CONFIG", "KUBECORE_RUN_ID",
            "KUBECORE_WORKFLOW_NAME", "KUBECORE_DATASET_DIR")

DATA_YAML = b"path: .\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: ship\n"


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


@pytest.fixture(autouse=True)
def clean_env(tmp_path, monkeypatch):
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("KAOS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("KUBECORE_CACHE_DIR", str(tmp_path / "cache"))  # deprecated alias
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def fake():
    f = FakeLakeFS()
    files = {"data.yaml": DATA_YAML, "images/train/a.jpg": b"img", "labels/train/a.txt": b"0 0.5 0.5 0.1 0.1\n",
             "README.md": b"notes"}
    f.add_repo("dataset-cats", metadata={"kaos.kind": "dataset", "kaos.template": "ultralytics"}, files=files)
    f.tag("dataset-cats", "v1")
    f.add_repo("outputs-yolo", metadata={"kaos.kind": "outputs"})
    return f


@pytest.fixture
def client(fake):
    return fake.client()


@pytest.fixture
def connected(fake, monkeypatch):
    """LakeFS.connect() returns the fake: for code that connects on its own."""
    from kaos_lib.lakefs import LakeFS
    monkeypatch.setattr(LakeFS, "connect", classmethod(lambda cls, *a, **k: fake.client()))
    return fake
