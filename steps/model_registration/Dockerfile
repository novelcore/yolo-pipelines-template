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
 # charset-normalizer repair: poetry lives in system site-packages
 # (virtualenvs.create false) and drags in a newer charset-normalizer whose
 # Cython-compiled extensions (cd/md .so, 3.5.0+) survive the locked
 # downgrade as ORPHANS no RECORD owns — compiled modules then shadow the
 # locked version's .py files -> "module 'charset_normalizer.md' has no
 # attribute 'CharInfo'" at runtime. --force-reinstall cannot remove
 # orphans, so: scorched-earth remove ALL remnants, clean-install the
 # exact locked version, and ASSERT the import at build time so this
 # class fails the build, never a training run.
 && CN_VER=$(grep -A1 'name = "charset-normalizer"' poetry.lock | sed -n 's/^version = "\(.*\)"/\1/p' | head -1) \
 && pip uninstall -y charset-normalizer \
 && rm -rf /usr/local/lib/python3.12/site-packages/charset_normalizer* \
 && pip install --no-cache-dir --no-deps "charset-normalizer==${CN_VER}" \
 && python -c "from charset_normalizer import md; import requests"

COPY app/ ./app/

# Argo runs: python -m app.entry --params <params> --training-result <json> ...
ENTRYPOINT ["python", "-m", "app.entry"]
