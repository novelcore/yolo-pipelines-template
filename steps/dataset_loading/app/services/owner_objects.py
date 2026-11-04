"""Object reads for a SHARED dataset, through the owner's lakeFS API (kaos-lib).

dataset-loading reads a dataset's objects through a boto3-style client: the consumer's own lakeFS
S3 gateway. A dataset another project shares (``<owner>/dataset-x@v1``) does not exist there, and
the consumer holds no S3 keys for the owner's lakeFS. The owner's lakeFS accepts the consumer's
token on its API only, so this adapter gives the service the three S3 calls it makes
(``head_object``, ``get_paginator("list_objects_v2")``, ``download_file``) on top of the
owner-aware kaos-lib client ``datasets.source`` returns.

Addressing is the one the service already uses: bucket = repository, key = ``<ref>/<path>``.
"""
from __future__ import annotations

import pathlib
from typing import Any, Iterator

_PAGE = 1000  # keys per listing page, as S3's list_objects_v2


class ObjectNotFound(Exception):
    """head_object on a key the owner's lakeFS does not hold (the service treats any error as absent)."""


def _split(key: str) -> tuple:
    ref, _, path = key.partition("/")
    return ref, path


class _Paginator:
    def __init__(self, client: Any) -> None:
        self._c = client

    def paginate(self, Bucket: str, Prefix: str = "") -> Iterator[dict]:  # noqa: N803 - boto3's names
        ref, path = _split(Prefix)
        page: list = []
        for obj in self._c.list_objects(Bucket, ref, path):
            page.append({"Key": f"{ref}/{obj['path']}", "Size": obj.get("size_bytes", 0)})
            if len(page) >= _PAGE:
                yield {"Contents": page}
                page = []
        if page:
            yield {"Contents": page}


class OwnerObjects:
    """The S3 calls of dataset-loading, served by a kaos-lib ``LakeFS`` client (the owner's)."""

    def __init__(self, client: Any) -> None:
        self._c = client

    def get_paginator(self, operation: str) -> _Paginator:
        if operation != "list_objects_v2":
            raise NotImplementedError(operation)
        return _Paginator(self._c)

    def head_object(self, Bucket: str, Key: str) -> dict:  # noqa: N803
        ref, path = _split(Key)
        for obj in self._c.list_objects(Bucket, ref, path):
            if obj["path"] == path:
                return {"ContentLength": obj.get("size_bytes", 0)}
        raise ObjectNotFound(f"s3://{Bucket}/{Key}")

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:  # noqa: N803
        ref, path = _split(Key)
        self._c.download_object(Bucket, ref, path, pathlib.Path(Filename))
