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
 && poetry install --only=main --no-root --no-interaction --no-ansi \
 # charset-normalizer repair: poetry itself lives in system site-packages
 # (virtualenvs.create false) and drags in a NEWER charset-normalizer whose
 # compiled md extension survives the locked downgrade and shadows the pure-
 # python module -> "module 'charset_normalizer.md' has no attribute
 # 'CharInfo'" at runtime (every fresh build since charset-normalizer 3.5.0
 # released). Force-reinstall the locked version so site-packages is coherent.
 && pip install --no-cache-dir --force-reinstall --no-deps "charset-normalizer==$(pip show charset-normalizer | sed -n 's/^Version: //p')"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <params> --training-result <json> ...
ENTRYPOINT ["python", "-m", "app.entry"]
