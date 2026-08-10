# model-registration step image — REAL registration (kubeline registrar vendored
# into app/). Deps via POETRY from the vendored pyproject.toml + poetry.lock, so
# versions are pinned identical to kubeline. Lighter than model-training (no torch).
# The platform builds this on push (kaniko -> Zot) and wires it as
# image-model-registration. In-image command: python -m app.entry --params <params>.
FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir poetry

COPY pyproject.toml poetry.lock* README.md ./
RUN poetry config virtualenvs.create false \
 && poetry install --only=main --no-root --no-interaction --no-ansi

COPY app/ ./app/

# Argo runs: python -m app.entry --params <params> --training-result <json> ...
ENTRYPOINT ["python", "-m", "app.entry"]
