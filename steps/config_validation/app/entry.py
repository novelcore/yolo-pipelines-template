"""config-validation — the Hera-way entry point.

The pipeline's first developer step: a pre-GPU gate that proves the run's
external dependencies exist before any expensive step starts. It reads the
single resolved params.yaml that compose-and-validate produced (--params) — the
same file every step receives — and runs liveness checks (dataset path / MLflow
/ pretrained weights / checkpoint) via app.manager.Manager.

Endpoints + credentials are read from the platform-injected env
(LAKEFS_ENDPOINT / LAKEFS_ACCESS_KEY / LAKEFS_SECRET_KEY, or the AWS_* chain,
and MLFLOW_TRACKING_URI). The developer wires none of that; checks whose
endpoint was not injected self-skip.
"""

import argparse
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version

import yaml

from app.manager import Manager

# The config sections this step reads from params.yaml. The compose-and-validate
# release gate fails the render if any of these sections is missing, so a
# renamed/deleted section is caught before any run.
READS = ["data", "dataset", "model"]


def kaos_lib_version() -> str:
    """The installed kaos-lib version (distribution `kaos-lib`, or `kubecore` before its first release)."""
    for dist in ("kaos-lib", "kubecore"):
        try:
            return package_version(dist)
        except PackageNotFoundError:
            pass
    return "unknown"


def main() -> None:
    print(f"kaos-lib {kaos_lib_version()}", flush=True)  # the package this image runs
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", required=True,
                        help="Resolved params.yaml content (from compose-and-validate).")
    args, _ = parser.parse_known_args()
    resolved = yaml.safe_load(args.params)

    Manager().run(resolved)


if __name__ == "__main__":
    main()

