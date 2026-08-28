"""Off-cluster lakeFS object I/O with the run's bearer token (PRD-1016 F-05).

In-cluster the step talks to the lakeFS *S3 gateway* with SigV4 keys
(LAKEFS_ACCESS_KEY / LAKEFS_SECRET_KEY). Off-cluster — a MeluXina Slurm job —
the platform hands the job only ``LAKEFS_ENDPOINT`` (public ingress) and
``LAKEFS_BEARER_TOKEN`` (a Zitadel JWT the ingress accepts); the S3 gateway is
not reachable that way, but the lakeFS *objects API* is. This module is the
bearer-mode twin of the boto3 calls the trainer makes: same ``bucket`` (the
lakeFS repository) and ``key`` (``{branch}/{path}``) vocabulary, stdlib only.
"""
from __future__ import annotations

import os
import urllib.parse
import urllib.request
from pathlib import Path


def bearer_available() -> bool:
    return bool(os.environ.get("LAKEFS_BEARER_TOKEN") and os.environ.get("LAKEFS_ENDPOINT"))


def _base() -> str:
    return os.environ["LAKEFS_ENDPOINT"].rstrip("/") + "/api/v1"


def _auth() -> dict:
    return {"Authorization": "Bearer " + os.environ["LAKEFS_BEARER_TOKEN"]}


def split_key(key: str) -> tuple[str, str]:
    """``{branch}/{path}`` -> (branch, path); the first segment is the ref."""
    branch, sep, path = key.partition("/")
    if not sep or not path:
        raise ValueError(
            f"lakeFS key {key!r} must be '{{branch}}/{{path}}' — the first segment is the branch"
        )
    return branch, path


def upload(local_path: Path, bucket: str, key: str, opener=urllib.request.urlopen) -> None:
    branch, path = split_key(key)
    data = Path(local_path).read_bytes()
    body = (
        b'--B\r\nContent-Disposition: form-data; name="content"; filename="f"\r\n'
        b"Content-Type: application/octet-stream\r\n\r\n" + data + b"\r\n--B--\r\n"
    )
    q = urllib.parse.urlencode({"path": path})
    req = urllib.request.Request(
        f"{_base()}/repositories/{bucket}/branches/{branch}/objects?{q}",
        data=body, method="POST",
        headers={**_auth(), "Content-Type": "multipart/form-data; boundary=B"},
    )
    with opener(req, timeout=600) as resp:
        resp.read()


def download(bucket: str, key: str, local_path: Path, opener=urllib.request.urlopen) -> None:
    ref, path = split_key(key)
    q = urllib.parse.urlencode({"path": path})
    req = urllib.request.Request(
        f"{_base()}/repositories/{bucket}/refs/{ref}/objects?{q}", headers=_auth()
    )
    with opener(req, timeout=600) as resp, open(local_path, "wb") as out:
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


class BearerClient:
    """Drop-in for the two boto3 calls the step services make
    (``download_file`` / ``upload_file``), backed by the objects API above.
    Built by ``Manager._build_s3_client`` when the platform hands the job a
    bearer token but no S3 gateway keys (MeluXina)."""

    def __init__(self, opener=urllib.request.urlopen):
        self._open = opener

    def download_file(self, bucket: str, key: str, local_path) -> None:
        download(bucket, key, Path(local_path), opener=self._open)

    def upload_file(self, local_path, bucket: str, key: str) -> None:
        upload(Path(local_path), bucket, key, opener=self._open)
