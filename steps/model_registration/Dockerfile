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

# The kubecore PACKAGE (its lakeFS client tags the registered model's commit in outputs-<app>),
# pinned to a commit of hera-pipeline-template, the same commit as the other steps.
# Bump the commit to take a newer package.
ARG KUBECORE_PKG_COMMIT=0136a205a82fbe4ed4a82037329e048f58e7e740
RUN pip install --no-cache-dir \
    "https://github.com/novelcore/hera-pipeline-template/archive/${KUBECORE_PKG_COMMIT}.zip#subdirectory=packages/kubecore"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <params> --training-result <json> ...
ENTRYPOINT ["python", "-m", "app.entry"]
