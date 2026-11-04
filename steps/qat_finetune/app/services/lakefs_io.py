"""lakeFS I/O of the ML steps, through kaos-lib (datasets and run outputs).

Three jobs, one place, and the file is IDENTICAL in model-training, qat-finetune and
model-quantization (each step image carries only its own app/):

- ``materialize``: the run's dataset at the commit dataset-loading pinned, as local files
  (the package's yolo consumer: one download, or the copy staged on MeluXina via
  ``KUBECORE_STAGED_DIR``). data.yaml comes back with ``path:`` already pointing at it.
- ``download_object``: one object of ``s3://<repo>/<ref>/<path>`` (a checkpoint, the qat tflite).
- ``run_writer`` / ``write_outputs``: the step's results through the package's run writer:
  saved on the branch ``run-<run>-<kind>`` of ``outputs-<app>`` (durable, reused on a retry), merged
  into ``main`` by ``finish`` as ONE commit under ``<kind>/<experiment>/<run>/``, branch deleted.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

from kaos_lib import datasets, outputs
from kaos_lib.formats import get_format
from kaos_lib.lakefs import LakeFS

OUTPUTS_REF = "main"  # the branch results are merged to; the first segment of the s3:// keys we return


def run_id() -> str:
    """The workflow name the platform injects as KUBECORE_RUN_ID ("local" off-platform)."""
    return os.environ.get("KUBECORE_RUN_ID", "").strip() or "local"


def materialize(dataset_ref: str, dataset_commit: str, dest: str,
                client: Optional[LakeFS] = None) -> Path:
    """Local data.yaml of ``dataset_ref`` at ``dataset_commit`` (copied/linked into ``dest``).

    Stops when the upstream step recorded no commit: reading a moving ref would let two
    steps of one run see different data.
    """
    if not (dataset_ref and dataset_commit):
        raise SystemExit("no pinned dataset: the upstream result carries no dataset_ref / "
                         "dataset_commit, so there is nothing to read.")
    ref = datasets.DatasetRef.parse(dataset_ref).with_commit(dataset_commit)
    return get_format("ultralytics").consumer.materialize(ref, dest, client)


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


def run_writer(app: str, experiment: str, kind: str = "checkpoints",
               client: Optional[LakeFS] = None) -> "outputs.RunWriter":
    """An OPEN run writer on ``outputs-<app>`` for this run (``save`` ... then ``finish``).

    The branch is created or reused, so a retried or preempted step resumes from what it saved.
    """
    return outputs.run(app, experiment, run_id(), kind, client).open()


def result_prefix(app: str, experiment: str, kind: str = "checkpoints") -> str:
    """``s3://outputs-<app>/main/<kind>/<experiment>/<run>/``: where finished results can be read back."""
    return f"s3://{outputs.outputs_repo(app)}/{OUTPUTS_REF}/{outputs.run_prefix(experiment, run_id(), kind)}"


def write_outputs(app: str, experiment: str, files: Iterable, kind: str = "checkpoints",
                  metadata: Optional[dict] = None, client: Optional[LakeFS] = None) -> str:
    """Save ``files`` and merge them into main of ``outputs-<app>`` as one commit; returns ``result_prefix``.

    ``metadata`` (dataset ref/commit, ...) rides on the merge commit. Raises when nothing could be
    written: a result that never reached lakeFS must fail the step.
    """
    w = run_writer(app, experiment, kind, client)
    w.save(list(files))
    w.finish(metadata)
    return result_prefix(app, experiment, kind)
