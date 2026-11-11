#!/usr/bin/env bash
# Re-runnable docs check: parameters page up to date, then mkdocs --strict. The build runs scripts/docs_check.py
# first (words, removed names, links across this guide, the kaos-lib guide and the hera guide), so a finding fails it.
#   scripts/check-docs.sh                        all checks
#   scripts/check-docs.sh --external             also fetch every external link
#   DOCS_CHECK_OFFLINE=1 scripts/check-docs.sh   no network for the links: this repository alone
# Creates/reuses .venv-docs. No CI needed: run it before you push.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [ ! -d .venv-docs ]; then python3 -m venv .venv-docs; fi
.venv-docs/bin/pip install -q -r docs/requirements.txt

echo "== parameters table matches config/ and the step models"
.venv-docs/bin/python scripts/gen-parameters.py --check

echo "== words, removed names, links; then mkdocs build --strict"
if [ "${1:-}" = "--external" ]; then export DOCS_CHECK_EXTERNAL=1; fi
NO_MKDOCS_2_WARNING=1 .venv-docs/bin/mkdocs build --strict -d "$(mktemp -d)"

echo "docs check OK"
