"""lakeFS I/O of the ML steps, through the kubecore package (datasets and run outputs).

Three jobs, one place, and the file is IDENTICAL in model-training, qat-finetune and
model-quantization (each step image carries only its own app/):

- ``materialize``: the run's dataset at the commit dataset-loading pinned, as local files
  (the package's ultralytics consumer: one download, or the copy staged on MeluXina via
  ``KUBECORE_STAGED_DIR``). data.yaml comes back with ``path:`` already pointing at it.
- ``download_object``: one object of ``s3://<repo>/<ref>/<path>`` (a checkpoint, the qat tflite).
- ``write_outputs``: the step's results as ONE commit on ``main`` of ``outputs-<app>``,
  under ``<kind>/<experiment>/<run>/`` (``kubecore.outputs.write_run``).
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterable, Optional

from kubecore import datasets, outputs
from kubecore.formats import get_format
from kubecore.lakefs import LakeFS

OUTPUTS_REF = "main"  # the branch results are merged to; the first segment of the s3:// keys we return


def run_id() -> str:
    """The workflow name the platform injects as KUBECORE_RUN_ID ("local" off-platform)."""
    return os.environ.get("KUBECORE_RUN_ID", "").strip() or "local"


def _pinned_client(commit: str, client: Optional[LakeFS] = None) -> LakeFS:
    """A lakeFS client whose refs all resolve to ``commit``.

    The consumer resolves the ref it is given, so this makes it read the commit
    dataset-loading pinned even when the run picked ``latest`` (the package has no
    ``commit=`` argument yet; drop this when it has).
    """
    c = client or LakeFS.connect()
    c.branch_commit = lambda repo, branch: commit
    c.tag_commit = lambda repo, tag: commit
    return c


def materialize(dataset_ref: str, dataset_commit: str, dest: str,
                client: Optional[LakeFS] = None) -> Path:
    """Local data.yaml of ``dataset_ref`` at ``dataset_commit`` (copied/linked into ``dest``).

    Stops when the upstream step recorded no commit: reading a moving ref would let two
    steps of one run see different data.
    """
    if not (dataset_ref and dataset_commit):
        raise SystemExit("no pinned dataset: the upstream result carries no dataset_ref / "
                         "dataset_commit, so there is nothing to read.")
    ref = datasets.DatasetRef.parse(dataset_ref)
    return get_format("ultralytics").consumer.materialize(
        ref, dest, _pinned_client(dataset_commit, client))


def download_object(uri: str, dest_dir: str, client: Optional[LakeFS] = None) -> str:
    """Download ``s3://<repo>/<ref>/<path>`` into ``dest_dir``; returns the local path.

    The first key segment is the lakeFS ref, as everywhere in this pipeline.
    """
    repo, _, key = uri[len("s3://"):].partition("/")
    ref, _, path = key.partition("/")
    local = Path(dest_dir) / Path(path).name
    local.parent.mkdir(parents=True, exist_ok=True)
    (client or LakeFS.connect()).download_object(repo, ref, path, local)
    return str(local)


def write_outputs(app: str, experiment: str, files: Iterable, kind: str = "checkpoints",
                  client: Optional[LakeFS] = None) -> str:
    """Write ``files`` as one commit on ``main`` of ``outputs-<app>`` under ``<kind>/<experiment>/<run>/``.

    Returns the ``s3://`` prefix (``s3://outputs-<app>/main/<kind>/<experiment>/<run>/``) the
    files can be read back from. Raises when nothing could be written: a result that never
    reached lakeFS must fail the step.
    """
    run = run_id()
    with tempfile.TemporaryDirectory() as tmp:
        for f in files:
            shutil.copy(f, Path(tmp) / Path(f).name)
        outputs.write_run(app, experiment, run, tmp, kind=kind, client=client)
    return f"s3://{outputs.outputs_repo(app)}/{OUTPUTS_REF}/{outputs.run_prefix(experiment, run, kind)}"
