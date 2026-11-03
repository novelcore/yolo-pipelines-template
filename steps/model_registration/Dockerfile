# model-registration step image — REAL registration (kubeline registrar vendored
# into app/). Deps via POETRY from the vendored pyproject.toml + poetry.lock, so
# versions are pinned identical to kubeline. Lighter than model-training (no torch).
# The platform builds this on push (kaniko -> Zot) and wires it as
# image-model-registration. In-image command: python -m app.entry --params <params>.
FROM python:3.12-slim

WORKDIR /app

# Dependency install — poetry NEVER touches the runtime python.
# History (kubecore-operator#1008 thread): installing poetry into system
# site-packages let poetry's own requests/charset-normalizer stack collide
# with the locked versions — orphaned compiled extensions survived every
# repair attempt and resurfaced on cold (cache-less) builds as
# "module 'charset_normalizer.md' has no attribute 'CharInfo'" at runtime.
# Now poetry lives in its own venv (/opt/poetry) and is used ONLY to export
# the lock; the runtime python gets exactly poetry.lock via plain pip.
# The final import assert makes any residue fail the BUILD, never a run.
RUN python -m venv /opt/poetry \
 && /opt/poetry/bin/pip install --no-cache-dir poetry poetry-plugin-export
COPY pyproject.toml poetry.lock* README.md ./
RUN /opt/poetry/bin/poetry export --only=main -f requirements.txt --output /tmp/requirements.txt \
 && pip install --no-cache-dir -r /tmp/requirements.txt \
 && rm -rf /opt/poetry /tmp/requirements.txt \
 && python -c "from charset_normalizer import md; import requests"

# kaos-lib, formerly the kubecore package (its lakeFS client tags the registered model's
# commit in outputs-<app>). The PLATFORM picks it: the CI passes --build-arg
# KAOS_LIB_PACKAGE=<requirement> on every build, inside the range in .kaos/compat.yaml
# (docs/kaos-lib-versions.md). The value is the whole pip requirement (name, extras,
# version or URL), so the platform can move it to a kaos-lib release with no change here.
# This default is only for a build on your computer: until kaos-lib's first release it is
# a commit of hera-pipeline-template, where the package is still named `kubecore`. It
# shares the import name `kubecore` with the vendored engine at the repo root, which is
# NOT in this image: only app/ is copied.
ARG KAOS_LIB_PACKAGE="kubecore @ https://github.com/novelcore/hera-pipeline-template/archive/0136a205a82fbe4ed4a82037329e048f58e7e740.zip#subdirectory=packages/kubecore"
RUN pip install --no-cache-dir "${KAOS_LIB_PACKAGE}"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <params> --training-result <json> ...
ENTRYPOINT ["python", "-m", "app.entry"]
