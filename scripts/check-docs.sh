#!/usr/bin/env bash
# For the maintainers of the template: parameters page up to date, the docs checks, then mkdocs --strict.
# The checks (words, removed names, links across this guide, the kaos-lib guide and the hera guide) are
# kaos-lib's docs_check.py; this repository holds no copy, so a pipeline created from the template is never checked.
#   scripts/check-docs.sh                         all checks
#   scripts/check-docs.sh --external              also fetch every external link
#   DOCS_CHECK_KAOS_LIB=<folder> scripts/...      use this kaos-lib checkout instead of downloading its trunk
#   DOCS_CHECK_HERA=<folder> scripts/...          the same for the hera template
#   DOCS_CHECK_OFFLINE=1 scripts/...              this repository alone; it prints what it skipped
# Creates/reuses .venv-docs. No CI needed: run it before you push.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "$(git remote get-url origin 2>/dev/null || true)" in
  *novelcore/yolo-pipelines-template|*novelcore/yolo-pipelines-template.git) ;;
  *) echo "check-docs: these checks are for the yolo-pipelines-template repository itself; a pipeline created from it has nothing to check here."
     exit 0 ;;
esac

if [ ! -d .venv-docs ]; then python3 -m venv .venv-docs; fi
.venv-docs/bin/pip install -q -r docs/requirements.txt

echo "== parameters table matches config/ and the step models"
.venv-docs/bin/python scripts/gen-parameters.py --check

echo "== words, removed names, links"
lib="${DOCS_CHECK_KAOS_LIB:-}"
if [ -z "$lib" ]; then
  lib="$(mktemp -d)"
  trap 'rm -rf "$lib"' EXIT
  if ! curl -fsSL --retry 3 https://codeload.github.com/kaos-io/kaos-lib/tar.gz/refs/heads/main | tar -xz -C "$lib" --strip-components=1; then
    echo "check-docs: cannot download kaos-io/kaos-lib@main, which holds the checks. Try again, or set DOCS_CHECK_KAOS_LIB to a checkout of it." >&2
    exit 1
  fi
fi
if [ ! -f "$lib/docs_check.py" ]; then
  echo "check-docs: $lib has no docs_check.py: it is not a kaos-lib checkout with the docs checks." >&2
  exit 1
fi
if [ "${1:-}" = "--external" ]; then export DOCS_CHECK_EXTERNAL=1; fi
DOCS_CHECK_KAOS_LIB="$lib" .venv-docs/bin/python "$lib/docs_check.py" "$PWD"

echo "== mkdocs build --strict"
NO_MKDOCS_2_WARNING=1 .venv-docs/bin/mkdocs build --strict -d "$(mktemp -d)"

echo "docs check OK"
