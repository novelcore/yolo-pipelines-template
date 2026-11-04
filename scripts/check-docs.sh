#!/usr/bin/env bash
# Re-runnable docs check: parameters page up to date, mkdocs --strict, internal links and anchors.
#   scripts/check-docs.sh              offline checks
#   scripts/check-docs.sh --external   also fetch every external link (needs network)
# Creates/reuses .venv-docs. No CI needed: run it before you push.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d .venv-docs ]; then python3 -m venv .venv-docs; fi
.venv-docs/bin/pip install -q -r docs/requirements.txt

echo "== parameters table matches config/ and the step models"
.venv-docs/bin/python scripts/gen-parameters.py --check

echo "== mkdocs build --strict"
.venv-docs/bin/mkdocs build --strict -d "$(mktemp -d)"

echo "== links and anchors"
.venv-docs/bin/python scripts/check-links.py "$@"

echo "docs check OK"
