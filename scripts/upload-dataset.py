#!/usr/bin/env python3
"""Upload a local YOLO-pose dataset to your app's lakeFS.

Thin wrapper around the vendored ``kubecore-dataset`` CLI (``dataset_tools/``).
Your pipeline's ``dataset_loading`` reads the dataset from the branch root::

    s3://<repo>/<branch>/dataset/     (images/..., labels/..., data.yaml)

so we log in (browser SSO), validate the dataset, and incrementally sync it
(uploads AND deletions) into ``dataset/`` on ``<branch>``, then commit.

    ./scripts/upload-dataset.py <local-dataset-dir> [version]
                                [--branch B] [--url URL] [--repo REPO] [--dry-run]

  <local-dataset-dir>  Directory with data.yaml + images/{train,val,test} +
                       labels/{train,val,test} at its ROOT (Ultralytics pose).
  version              Provenance tag recorded in MLflow. Defaults to the branch.

lakeFS URL + repo + branch are read from .kubecore/dataset-config.yaml when it
exists, so you can pass only the dataset dir. Otherwise pass --url and --repo.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

# Make the vendored dataset_tools importable whether run from the repo root or
# elsewhere.
_REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "dataset_tools"))

from dataset_cli.appconfig import config_value  # noqa: E402
from dataset_cli.lakefs_client import LakeFSClient  # noqa: E402
from dataset_cli.login import login as do_login  # noqa: E402
from dataset_cli.sync import sync as do_sync  # noqa: E402
from dataset_cli.validate import validate_dataset  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Upload a YOLO-pose dataset to lakeFS (this app's layout).")
    ap.add_argument("dataset_dir", help="local dataset directory (Ultralytics pose layout at root)")
    ap.add_argument("version", nargs="?", default=None,
                    help="MLflow provenance tag (default: the branch)")
    ap.add_argument("--branch", default=None, help="lakeFS branch / data-ref (default: config or 'main')")
    ap.add_argument("--url", default=None, help="lakeFS ingress URL (default: .kubecore/dataset-config.yaml)")
    ap.add_argument("--repo", default=None, help="lakeFS repo (default: .kubecore/dataset-config.yaml)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--prefix", default=None,
                    help="object prefix under the branch (default: dataset)")
    ap.add_argument("--dry-run", action="store_true", help="show what would sync, upload nothing")
    args = ap.parse_args()

    start = args.dataset_dir
    url = args.url or config_value("lakefsUrl", start)
    repo = args.repo or config_value("repo", start)
    branch = args.branch or config_value("branch", start) or "main"
    version = args.version or branch
    prefix = args.prefix.strip("/") if args.prefix else "dataset"

    if not url or not repo:
        sys.exit("ERROR: could not determine lakeFS url/repo. Run inside the app "
                 "clone (it ships .kubecore/dataset-config.yaml), or pass --url/--repo.")

    # 1) validate locally before touching the network
    result = validate_dataset(args.dataset_dir)
    if not result.ok:
        print("✗ Dataset invalid — fix these before uploading:")
        for e in result.errors:
            print(f"    - {e}")
        return 2
    print(f"✓ Dataset valid. Target: s3://{repo}/{branch}/{prefix}/")

    if args.dry_run:
        print("(--dry-run) skipping login + upload.")
        return 0

    # 2) browser SSO login -> session cookie
    cookie = do_login(url)
    client = LakeFSClient(url, cookie, concurrency=args.concurrency)
    if not client.branch_exists(repo, branch):
        client.ensure_branch(repo, branch, client.default_branch(repo))

    # 3) incremental sync into dataset/ (uploads + deletions), then commit
    commit_id = do_sync(pathlib.Path(args.dataset_dir), client, repo, branch,
                        prefix=prefix, concurrency=args.concurrency)
    print(f"\n✓ Uploaded to s3://{repo}/{branch}/{prefix}/  (commit {commit_id})")
    print(f"  Run the pipeline with data-ref={branch}"
          + (f" and data-version={version}" if version != branch else "") + ".")
    return 0


if __name__ == "__main__":
    sys.exit(main())
